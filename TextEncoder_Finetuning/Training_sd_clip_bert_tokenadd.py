# train_sdfusion_bert_lora.py
# -*- coding: utf-8 -*-
import os, math, json, sys
sys.path.append(r"D:\Junyhuang\Project2\ControlNet")
import matplotlib.pyplot as plt

# hard-disable the checkpoint wrapper everywhere
def _no_checkpoint(func, inputs, params=None, flag=None):
    # Run forward as-is, no recomputation checkpointing
    return func(*inputs)

# Patch util.checkpoint
import importlib
ldm_util = importlib.import_module("ldm.modules.diffusionmodules.util")
ldm_util.checkpoint = _no_checkpoint

# Patch the *name* 'checkpoint' inside attention.py as well (it does: from util import checkpoint)
ldm_attn = importlib.import_module("ldm.modules.attention")
ldm_attn.checkpoint = _no_checkpoint

print("[patch] Disabled ldm checkpoint: util.checkpoint AND attention.checkpoint are now no-ops.")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from ControlNet.ldm.util import instantiate_from_config
from torch.amp import GradScaler, autocast
from transformers import BertModel, BertTokenizer
from token_utils import build_cond_uc_safe
from token_utils import BertAlign, TokenAlign77

import ControlNet.ldm.modules.diffusionmodules.util as ldm_util

def _no_checkpoint(func, inputs, params, flag):
    # just run forward without any checkpointing
    return func(*inputs)

ldm_util.checkpoint = _no_checkpoint
print("[patch] ldm.util.checkpoint disabled at runtime.")

CKPT    = r"D:\Junyhuang\Project2\BaseModel\Swisstopo.ckpt"   # whole ckpt include U-Net
CFG     = r"D:\Junyhuang\Project2\ControlNet\models\cldm_v15.yaml"
ROOTDIR = r"D:\Junyhuang\Project2_Data\Training Data\Item_color"
JSONL   = os.path.join(ROOTDIR, "meta", "pairs.jsonl")
SPLIT_DIR = os.path.join(ROOTDIR, "meta_splits_small_train_sample")
OUTDIR  = r"D:\Junyhuang\Project2\Outputs\sdfusion_bert_ll_ofsp_color\adpter_LoRA_add_tokens"
LOSS_LOG_PATH = os.path.join(OUTDIR, "vis", "loss_log.json")
os.makedirs(os.path.join(OUTDIR, "vis"), exist_ok=True)

# Training hyperparameters
BATCH=8; SIZE=512; STEPS=2100; MAXLEN=77
USE_LORA = True
LORA_R=8     # lr is setting in later before build opt
VAL_EVERY=100
TEXT_DROP_PROB = 0.2   # 20% empty prompt possibility in training stage to force model learn to use text
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUTDIR, exist_ok=True)

# introduce tools from other script
from data_utils import stratified_split_ratio, PairsJSONLDataset
from vis_metrics import MeterEMA, validate_step, validate_image_metrics, sample_preview_new, save_ab_grid
SAVE_IMG_EVERY = 300    # steps for visualization

# introduce SDFusion text encoder
from SDFusion_bert.bert_network.network import BERTTextEncoder

# loss visualization for training insight
def _load_loss_log(path):
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                # basic field
                for k in ["step", "train_mse", "ema", "val_step", "val_mse", "train_l1", "val_ssim"]:
                    data.setdefault(k, [])
                return data
            except Exception:
                pass
    return {"step": [], "train_mse": [], "ema": [], "val_step": [], "val_mse": [], "train_l1": [], "val_ssim": []}

def _save_loss_log(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)

def _save_loss_plot(data, out_png):
    # plot the training loss curve
    plt.figure(figsize=(7.0, 4.0))
    if data["step"]:
        plt.plot(data["step"], data["train_mse"], label="train_mse", linewidth=1.2)
        if any(x is not None for x in data["ema"]):
            plt.plot(data["step"], data["ema"], label="ema", linewidth=1.2)
    if data["val_step"]:
        plt.plot(data["val_step"], data["val_mse"], "o-", markersize=3, label="val_mse")
    plt.xlabel("global_step")
    plt.ylabel("MSE")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def build_text_encoder(n_embed=768, n_layer=12, max_len=77, device="cuda", use_tokenizer=True):
    """
    Build SDFusion's BERTTextEncoder, return tokens [B, L, n_embed]。
    Default to froze layers and only train a adapter.
      tokens = Adapter( last_hidden_state )
    """
    print("[Init] Loading Hugging Face pretrained weights from 'bert-base-uncased'...")

    hf_bert = BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    hf_bert.to(device)

    # froze all parameters from hugging face bert textencoder
    for p in hf_bert.parameters():
        p.requires_grad = False

    # build a small adapter：LN -> Linear(768->768, initial zeros) -> sigmoid controller -> residual to input
    class LastStateAdapter(nn.Module):
        def __init__(self, dim=768, gate_init=-0.5):
            super().__init__()
            self.ln = nn.LayerNorm(dim)
            self.proj = nn.Linear(dim, dim, bias=True)
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
            self.beta = nn.Parameter(torch.tensor(gate_init))  # sigmoid(-0.5)≈0.38, upscale already (small start, no learn
        def forward(self, x):  # x: (B, L, 768)
            y = self.proj(self.ln(x))
            y = torch.sigmoid(self.beta) * y
            return x + y  # smaller residual

    adapter = LastStateAdapter(dim=n_embed).to(device)

    te = BERTTextEncoder(
        n_embed=n_embed, n_layer=n_layer, max_seq_len=max_len,
        vocab_size=30522, device=device, use_tokenizer=use_tokenizer, embedding_dropout=0.0
    ).to(device)

    # add adapter
    te.hf_bert = hf_bert
    te.tokenizer = tokenizer
    te.max_len = max_len
    te.adapter = adapter

    # return Adapter(last_hidden_state)
    def encode_on_device(prompts):
        enc = tokenizer(
            prompts, return_tensors='pt',
            padding="max_length", truncation=True, max_length=max_len
        )
        input_ids = enc['input_ids'].to(device)
        attn_mask = enc['attention_mask'].to(device)
        te.hf_bert.eval()
        with torch.no_grad():
            out = te.hf_bert(input_ids=input_ids, attention_mask=attn_mask, return_dict=True)
            tokens = out.last_hidden_state  # (B,77,768)
        tokens = te.adapter(tokens)        # only layer to train
        return tokens

    te.encode = encode_on_device

    trainable = sum(p.numel() for p in te.adapter.parameters() if p.requires_grad)
    print(f"[Init] BERTTextEncoder ready. Trainable adapter params: {trainable}")
    return te

# Adding LoRA to UNet cross-attn k/v
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=16, alpha=1.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        in_f, out_f = base.in_features, base.out_features
        dev = base.weight.device
        dty = base.weight.dtype

        # A/B create on the same device & dtype with base
        self.A = nn.Linear(in_f, r, bias=False).to(device=dev, dtype=dty)
        self.B = nn.Linear(r, out_f, bias=False).to(device=dev, dtype=dty)

        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

        self.scaling = (alpha if alpha is not None else r) / r

    def forward(self, x):
        # x and base/A/B in same device/dtype
        return self.base(x) + self.B(self.A(x)) * self.scaling


def lora_kv(unet, r=2):
    n = 0
    for m in unet.modules():
        if hasattr(m, "to_k") and isinstance(m.to_k, nn.Linear):
            m.to_k = LoRALinear(m.to_k, r=r); n += 1
        if hasattr(m, "to_v") and isinstance(m.to_v, nn.Linear):
            m.to_v = LoRALinear(m.to_v, r=r); n += 1
    return n


def ensure_lora_to_device(unet, device):
    for m in unet.modules():
        if isinstance(getattr(m, "to_k", None), LoRALinear):
            dev = m.to_k.base.weight.device; dty = m.to_k.base.weight.dtype
            m.to_k.A.to(device=dev, dtype=dty); m.to_k.B.to(device=dev, dtype=dty)
        if isinstance(getattr(m, "to_v", None), LoRALinear):
            dev = m.to_v.base.weight.device; dty = m.to_v.base.weight.dtype
            m.to_v.A.to(device=dev, dtype=dty); m.to_v.B.to(device=dev, dtype=dty)


def select_score(val_mse, val_l1, val_ssim, mode="l1", alpha=0.5):
    if mode == "l1":
        return val_l1  # smaller -> better
    if mode == "combo":
        return alpha * val_mse + (1.0 - alpha) * val_l1
    if mode == "ssim":
        return 1.0 - val_ssim  # SSIM larger -> better, change to smaller
    return val_mse


def disable_unet_checkpointing(unet):
    if hasattr(unet, "use_checkpoint"):
        unet.use_checkpoint = False
    for m in unet.modules():
        if hasattr(m, "checkpoint"):
            m.checkpoint = False

def assert_frozen_controlnet(model):
    bad = []
    for n,p in model.named_parameters():
        if (n.startswith("control_model.") or n.startswith("control_")) and p.requires_grad:
            bad.append(n)
    if bad:
        raise RuntimeError(f"[FROZEN CHECK] These ControlNet params are trainable but should be frozen: {bad}")
    print("[FROZEN CHECK] ControlNet parameters are all frozen.")


# 1. build model and load ckpt
cfg = OmegaConf.load(CFG)
model = instantiate_from_config(cfg.model)
sd = torch.load(CKPT, map_location="cpu"); sd = sd.get("state_dict", sd)
missing, unexpected = model.load_state_dict(sd, strict=False)
print("loaded ckpt. missing:", len(missing), "unexpected:", len(unexpected))

# move to GPU FIRST
model = model.to(device).eval()

# frozen: VAE / UNet / ControlNet
for p in model.first_stage_model.parameters(): p.requires_grad=False
for p in model.model.parameters(): p.requires_grad=False
for n,p in model.named_parameters():
    if n.startswith("control_model.") or n.startswith("control_"):
        p.requires_grad=False

assert_frozen_controlnet(model)

# LoRA trigger
if USE_LORA:
    wrapped = lora_kv(model.model.diffusion_model, r=LORA_R)
    print("LoRA-wrapped linears:", wrapped)
    ensure_lora_to_device(model.model.diffusion_model, device)
else:
    print("LoRA disabled: UNet to_k/to_v remain vanilla Linear.")

disable_unet_checkpointing(model.model.diffusion_model)
print("UNet gradient checkpointing disabled.")

try:
    model.model.diffusion_model.set_use_memory_efficient_attention_xformers(True)
    print("Enabled xformers memory efficient attention.")
except Exception as e:
    print("xformers not available:", e)

ensure_lora_to_device(model.model.diffusion_model, device)
for n,p in model.model.diffusion_model.named_parameters():
    if n.endswith("to_k.A.weight"):
        print("sample LoRA param device/dtype:", p.device, p.dtype)
        break

# text encoder
textenc = build_text_encoder(n_embed=768, n_layer=12, max_len=MAXLEN, device=device)
textenc.train()

bert_len_align = TokenAlign77(target_len=77).to(device)
bert_align     = BertAlign(dim=768, gate_init=-2.0).to(device, dtype=next(model.parameters()).dtype)

param_groups = []

# when add LoRA(K/V)
lora_params = [p for n,p in model.model.diffusion_model.named_parameters()
               if (".A.weight" in n or ".B.weight" in n) and p.requires_grad]
if USE_LORA and len(lora_params) > 0:
    param_groups.append({"params": lora_params, "lr": 1e-4, "weight_decay": 0.0})

param_groups.append({"params": textenc.adapter.parameters(),"lr": 6e-5, "weight_decay": 0.0})
param_groups.append({"params": bert_align.parameters(), "lr": 6e-5, "weight_decay": 0.0})

opt = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)

scaler = GradScaler(enabled=(device=="cuda"))

# lr dispatch: warmup + cosine
WARMUP = 500
TOTAL_STEPS = STEPS

BASE_LR = [g["lr"] for g in opt.param_groups]   # only use for print
WARMUP  = 500
MIN_LR  = 1e-6
def set_warmup_lr(step):
    for g, base in zip(opt.param_groups, BASE_LR):
        g["lr"] = base * min(1.0, step / WARMUP)

def cosine_decay_lr(step, total_steps=STEPS):
    t = (step - WARMUP) / max(1, (total_steps - WARMUP))
    t = min(max(t, 0.0), 1.0)
    for g, base in zip(opt.param_groups, BASE_LR):
        g["lr"] = MIN_LR + 0.5 * (base - MIN_LR) * (1.0 + math.cos(math.pi * t))


# 2. data split and build DataLoader
os.makedirs(SPLIT_DIR, exist_ok=True)
split_paths = stratified_split_ratio(JSONL, SPLIT_DIR, per_prompt_ratio=(0.1, 0.1, 0.8), min_per_split=(1, 1, 1), shuffle_seed=42, verbose=True)
train_jsonl = split_paths["train"]
val_jsonl   = split_paths["val"]
test_jsonl  = split_paths["test"]  # only use for evaluation

dl     = DataLoader(PairsJSONLDataset(train_jsonl, ROOTDIR, image_size=SIZE, seg_mode="RGB"),
                    batch_size=BATCH, shuffle=True,  num_workers=0, pin_memory=True)
dl_val = DataLoader(PairsJSONLDataset(val_jsonl,   ROOTDIR, image_size=SIZE, seg_mode="RGB"),
                    batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)

torch.backends.cudnn.benchmark = True
train_meter = MeterEMA(beta=0.98)
best_val = float("inf")
best_score = float("inf")
BEST_SELECTOR = "l1"
ALPHA = 0.5

loss_log = _load_loss_log(LOSS_LOG_PATH)

# check LoRA Frozen before training
trainable_lora = [n for n,p in model.model.diffusion_model.named_parameters()
                  if (".A.weight" in n or ".B.weight" in n) and p.requires_grad]
print(f"[check] trainable LoRA params: {len(trainable_lora)}")

# 3. training loop
global_step = 0
while global_step < STEPS:
    for batch in dl:
        global_step += 1

        if global_step == 1:
            print("adapter trainable? ", any(p.requires_grad for p in textenc.adapter.parameters()))
            print("num adapter params:", sum(p.numel() for p in textenc.adapter.parameters()))
            for i, g in enumerate(opt.param_groups):
                print(f"[pg{i}] lr={g['lr']}, n_params={sum(p.numel() for p in g['params'])}")

        seg = batch["seg"].to(device=device, dtype=torch.float32)
        gt  = batch["gt"].to(device=device, dtype=torch.float32)
        prompts = batch["prompt"]

        with torch.no_grad():
            z = model.get_first_stage_encoding(model.encode_first_stage(gt))  # [B,4,H/8,W/8]

        z = z.to(torch.float16)

        t = torch.randint(0, model.num_timesteps, (z.size(0),), device=device).long()
        noise  = torch.randn_like(z)
        z_noisy= model.q_sample(z, t, noise=noise)

        # z_noisy.requires_grad_(True)

        with autocast(device_type='cuda', enabled=(device == "cuda"), dtype=torch.float16):
            # keep Swisstopo original style embedding joint my BERT embedding
            # if dataloader [0,1]:
            seg = seg * 2.0 - 1.0
            cond, uc = build_cond_uc_safe(
                seg=seg, prompts=prompts,
                model=model, clip_textenc=model, bert_textenc=textenc, device=device,
                clip_style_text="map in swisstopo style",
                bert_len_align=bert_len_align, bert_align=bert_align,
            )

            # 正常的扩散训练：预测噪声
            eps_hat = model.apply_model(z_noisy, t, cond)
            loss = F.mse_loss(eps_hat.float(), noise.float())

            ema = train_meter.update(loss.item())
            loss_log["step"].append(int(global_step))
            loss_log["train_mse"].append(float(loss.item()))
            loss_log["ema"].append(float(ema))

        # opt.zero_grad(set_to_none=True)
        # Learning rate schedule
        if global_step <= WARMUP:
            set_warmup_lr(global_step)
        else:
            cosine_decay_lr(global_step)

        # Backpropagation with gradient scaling
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        # Unscale + gradient clipping
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(
            [p for g in opt.param_groups for p in g["params"] if p.grad is not None],
            max_norm=1.0
        )

        # Optimizer step
        scaler.step(opt)
        scaler.update()

        if global_step % 100 == 0:
            print(f"\[{global_step}/{STEPS}\] train_mse={loss.item():.4f}  ema={ema:.4f}")
            # adapter 的门控系数（sigmoid(beta) 越大，BERT 增量越强）
            if hasattr(textenc, "adapter") and hasattr(textenc.adapter, "beta"):
                beta_val = float(torch.sigmoid(textenc.adapter.beta).detach().cpu())
                print(f"adapter.gate(sigmoid(beta)) = {beta_val:.3f}")
            lr_str = ", ".join([f"{g['lr']:.1e}" for g in opt.param_groups])
            gate = float(torch.sigmoid(textenc.adapter.beta).detach())
            print(f"[{global_step}] train_mse={loss.item():.4f}  ema={ema:.4f}  lrs=[{lr_str}]  gate={gate:.3f}")

            # 验证 LoRA 确实在更新（梯度范数非零）
            def grad_norm_from_group(group):
                s = 0.0
                for p in group["params"]:
                    if p.grad is not None:
                        s += p.grad.data.pow(2).sum().item()
                return (s ** 0.5)

                # 默认 param_groups[0] = LoRA, param_groups[1] = adapter
            if USE_LORA:
                print("[grad] LoRA   =", f"{grad_norm_from_group(opt.param_groups[0]):.3e}")
            print("[grad] adapter =", f"{grad_norm_from_group(opt.param_groups[1]):.3e}")

            # 第一次打印 seg 的数值域，确认是否需要 seg=seg*2-1
            if global_step == 100:
                smin = float(seg.min()); smax = float(seg.max()); smean = float(seg.mean())
                print(f"[debug] seg range: min={smin:.3f}  max={smax:.3f}  mean={smean:.3f}")

        if global_step % VAL_EVERY == 0:
            model.eval(); textenc.eval()

            val_iter = iter(dl_val)
            try:
                vb0 = next(val_iter)
            except StopIteration:
                vb0 = next(iter(dl_val))

            gt0 = vb0["gt"].to(device=device, dtype=torch.float32)
            with torch.no_grad():
                z0 = model.get_first_stage_encoding(model.encode_first_stage(gt0)).to(dtype=torch.float16)

            torch.manual_seed(42)
            fixed_t = torch.randint(0, model.num_timesteps, (z0.size(0),), device=device).long()
            fixed_noise = torch.randn(z0.shape, device=z0.device, dtype=z0.dtype)
            def validate_step_once(model, textenc, batch, device, fixed_t, fixed_noise):
                seg = batch["seg"].to(device=device, dtype=torch.float32)
                gtb = batch["gt"].to(device=device, dtype=torch.float32)
                B, _, H, W = seg.shape
                with torch.no_grad():
                    z = model.get_first_stage_encoding(model.encode_first_stage(gtb)).to(dtype=fixed_noise.dtype)
                    t = fixed_t[:B]
                    n = fixed_noise[:B]
                    z_noisy = model.q_sample(z, t, noise=n)

                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(device == "cuda")):
                    # 若你的 ControlNet 训练期 hint 是 [-1,1]、而 seg 为 [0,1]，打开下一行
                    seg = seg * 2.0 - 1.0
                    cond, _uc = build_cond_uc_safe(
                        seg=seg, prompts=batch["prompt"],
                        model=model, clip_textenc=model, bert_textenc=textenc, device=device,
                        clip_style_text="map in swisstopo style",
                        bert_len_align=bert_len_align, bert_align=bert_align,
                    )
                    eps_hat = model.apply_model(z_noisy, t, cond)
                    loss = F.mse_loss(eps_hat.float(), n.float())
                return loss.item()


            num_batches = min(8, len(dl_val))
            val_iter = iter(dl_val)
            val_sum = 0.0
            for _ in range(num_batches):
                try:
                    val_batch = next(val_iter)
                except StopIteration:
                    break
                bs = val_batch["gt"].size(0)
                val_sum += validate_step_once(
                    model, textenc, val_batch, device,
                    fixed_t=fixed_t[:bs], fixed_noise=fixed_noise[:bs]
                )
            val_mse = val_sum / max(1, num_batches)

            # ---- img matrics（L1/SSIM）----
            img_metrics = validate_image_metrics(
                model, textenc, vb0, device,
                steps=15, scale=4.5, eta=0.0
            )
            val_l1, val_ssim = img_metrics["l1"], img_metrics["ssim"]

            print(f"   -> val_mse={val_mse:.4f}  val_l1={val_l1:.4f}  val_ssim={val_ssim:.3f}")

            loss_log["val_step"].append(int(global_step))
            loss_log["val_mse"].append(float(val_mse))
            loss_log["train_l1"].append(float(val_l1))
            loss_log["val_ssim"].append(float(val_ssim))

            # viz: A(my prompt) / B(keep prompt) ----
            if global_step % SAVE_IMG_EVERY == 0:
                take = min(batch["seg"].size(0), 2)
                seg_a = batch["seg"][:take].detach()
                gt_a = batch["gt"][:take].detach()
                prom_a = batch["prompt"][:take]

                preds_a = sample_preview_new(
                    model, textenc,
                    {"seg": seg_a, "gt": gt_a, "prompt": prom_a},
                    device, steps=20, scale=4.5, eta=0.0, seed=1234
                )
                preds_b = sample_preview_new(
                    model, textenc,
                    {"seg": seg_a, "gt": gt_a, "prompt": ["keep original color."] * take},
                    device, steps=20, scale=4.5, eta=0.0, seed=1234
                )

                out_png = os.path.join(OUTDIR, f"vis/train_ab_{global_step:06d}.png")
                save_ab_grid(seg_a, preds_a, preds_b, gt_a, out_png, prompts_a=prom_a)
                print(f"   -> saved TRAIN A/B preview to {out_png}")

            _save_loss_log(LOSS_LOG_PATH, loss_log)
            _save_loss_plot(loss_log, os.path.join(OUTDIR, "vis", "loss_curve.png"))

            # 保存最优（只保存 adapter + LoRA）
            score = select_score(val_mse, val_l1, val_ssim, mode=BEST_SELECTOR, alpha=ALPHA)
            if score < best_score:
                best_score = score
                # 只保存 adapter
                torch.save(textenc.adapter.state_dict(), os.path.join(OUTDIR, "best_text_adapter.pt"))
                # # 保存 LoRA（如果有）
                lora_sd = {k: v for k, v in model.model.diffusion_model.state_dict().items()
                           if k.endswith(".A.weight") or k.endswith(".B.weight")}
                if len(lora_sd) > 0:
                    torch.save(lora_sd, os.path.join(OUTDIR, "best_unet_lora_kv.pt"))
                print(f"   -> saved BEST (selector={BEST_SELECTOR}, score={best_score:.4f}) at step {global_step}")

            model.train(); textenc.train()

        if global_step >= STEPS:
            break

# save current ckpt at the end
torch.save(textenc.adapter.state_dict(), os.path.join(OUTDIR, "last_text_adapter.pt"))
lora_sd = {k: v for k, v in model.model.diffusion_model.state_dict().items()
           if k.endswith(".A.weight") or k.endswith(".B.weight")}
torch.save(lora_sd, os.path.join(OUTDIR, "last_unet_lora_kv.pt"))
print("done.")

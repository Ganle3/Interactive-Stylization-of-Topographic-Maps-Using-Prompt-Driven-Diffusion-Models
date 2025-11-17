# multi seg multi targets: forest (light/dark green), buildings (gray), road (gray), river (steel blue) in diff tile
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
import random, numpy as np
from torch.utils.data import Subset

import ControlNet.ldm.modules.diffusionmodules.util as ldm_util

def _no_checkpoint(func, inputs, params, flag):
    # just run forward without any checkpointing
    return func(*inputs)

ldm_util.checkpoint = _no_checkpoint
print("[patch] ldm.util.checkpoint disabled at runtime.")

CKPT    = r"D:\Junyhuang\Project2\BaseModel\Swisstopo.ckpt"   # whole ckpt include U-Net
CFG     = r"D:\Junyhuang\Project2\ControlNet\models\cldm_v15.yaml"
ROOTDIR = r"D:\Junyhuang\Project2_Data\Training Data\Sameseg_diffitem_8"
JSONL   = os.path.join(ROOTDIR, "meta", "pairs_seggroup.jsonl")
SPLIT_DIR = os.path.join(ROOTDIR, "meta_overfit_test_sample")
OUTDIR  = r"D:\Junyhuang\Project2\Outputs_overfit\Sameseg_multiprompt\batch_switch_lora_Com"
LOSS_LOG_PATH = os.path.join(OUTDIR, "vis", "loss_log.json")
os.makedirs(os.path.join(OUTDIR, "vis"), exist_ok=True)

# Training hyperparameters
BATCH=8; SIZE=512; STEPS=15000; MAXLEN=77
LORA_R=12     # lr is setting in later before build opt
VAL_EVERY=500
TEXT_DROP_PROB = 0.0   # empty prompt possibility in training stage to force model learn to use text
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUTDIR, exist_ok=True)

# introduce tools from other script
from TextEncoder_Finetuning.data_utils import stratified_split_ratio, PairsJSONLDataset
from TextEncoder_Finetuning.vis_metrics import MeterEMA, validate_step, save_concept_grid_new, sample_preview_CNLora
SAVE_IMG_EVERY = 500    # steps for visualization

# introduce SDFusion text encoder
from SDFusion_bert.bert_network.network import BERTTextEncoder


# loss visualization for training insight
def _load_loss_log(path):
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                # basic field
                for k in ["step", "train_mse", "ema", "val_step", "val_mse", "val_l1", "val_ssim"]:
                    data.setdefault(k, [])
                return data
            except Exception:
                pass
    return {"step": [], "train_mse": [], "ema": [], "val_step": [], "val_mse": [], "val_l1": [], "val_ssim": []}

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

# === 关闭 BERT Dropout，方便单样本过拟合调试 ===
def disable_bert_dropout(hf_bert):
    """将 HuggingFace BERT 模型的所有 dropout 层关闭。"""
    # 修改 config 层面的概率
    if hasattr(hf_bert, "config"):
        hf_bert.config.hidden_dropout_prob = 0.0
        hf_bert.config.attention_probs_dropout_prob = 0.0

    # 修改模块实例里的 Dropout 层
    import torch.nn as nn
    for m in hf_bert.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0

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

    # # build a small adapter：LN -> Linear(768->768, initial zeros) -> sigmoid controller -> residual to input
    # class LastStateAdapter(nn.Module):
    #     def __init__(self, dim=768, gate_init=-0.5):
    #         super().__init__()
    #         self.ln = nn.LayerNorm(dim)
    #         self.proj = nn.Linear(dim, dim, bias=True)
    #         nn.init.zeros_(self.proj.weight)
    #         nn.init.zeros_(self.proj.bias)
    #         self.beta = nn.Parameter(torch.tensor(gate_init))  # sigmoid(-0.5)≈0.38, upscale already (small start, no learn
    #     def forward(self, x):  # x: (B, L, 768)
    #         y = self.proj(self.ln(x))
    #         y = torch.sigmoid(self.beta) * y
    #         return x + y  # smaller residual
    # LN → Linear → GELU → Linear
    class LastStateAdapter(nn.Module):
        def __init__(self, dim=768, hidden=768):
            super().__init__()

            self.ff = nn.Sequential(
                nn.Linear(dim, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),

                # nn.Linear(hidden, hidden),
                # nn.ReLU(),
                # nn.LayerNorm(hidden),

                nn.Linear(hidden, dim, bias=True),
            )
            self.final_ln = nn.LayerNorm(dim) # mimics CLIP's final LayerNorm

        def forward(self, x):
            y = self.ff(x)
            y = self.final_ln(y)
            return y

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
    """
    - 对 base Linear 做 LoRA（A,B）
    - 允许外部注入 seg_feat/text_feat（通过 set_control_feature）
    - 融合层 fuse_mlp 懒初始化：第一次拿到 seg/text 后，根据真实维度建
    - forward: out = base(x) + B(A(x + Δx)) * scaling，其中 Δx 由 [seg,text] 映射到 in_features 形成
    """
    def __init__(self, base: nn.Linear, r=16, alpha=None):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        in_f, out_f = base.in_features, base.out_features
        dev = base.weight.device
        dty = base.weight.dtype

        # 仅首次打印一次尺寸，便于确认
        if not hasattr(self.__class__, "_printed"):
            print(f"[LoRALinear] Inject on Linear: in={in_f}, out={out_f}")
            self.__class__._printed = True

        # LoRA 权重
        self.A = nn.Linear(in_f, r, bias=False).to(device=dev, dtype=dty)
        self.B = nn.Linear(r, out_f, bias=False).to(device=dev, dtype=dty)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)
        self.scaling = (alpha or (2 * r)) / r

        # 条件特征（懒初始化）
        self.seg_feat = None      # [B, C_seg]
        self.text_feat = None     # [B, C_txt]
        self.fuse_mlp = None      # 在第一次收到 seg/text 后按真实维度构建
        self._shape_printed = False

    @torch.no_grad()
    def set_control_feature(self, seg_feat: torch.Tensor, text_tokens: torch.Tensor = None):
        """
        seg_feat: [B, C_seg]   （你可以用自适应池化或别的方式得到）
        text_tokens: [B, L, C_txt] 或 [B, C_txt]
        """
        dev = self.base.weight.device
        dty = self.base.weight.dtype

        # 规范到 2D
        if seg_feat is not None and seg_feat.ndim > 2:
            seg_feat = seg_feat.view(seg_feat.size(0), -1)
        self.seg_feat = seg_feat.to(device=dev, dtype=dty) if seg_feat is not None else None

        if text_tokens is not None:
            if text_tokens.ndim == 3:
                text_feat = text_tokens.mean(dim=1)  # [B, L, C] → [B, C]
            else:
                text_feat = text_tokens
            self.text_feat = text_feat.to(device=dev, dtype=dty)
        else:
            self.text_feat = None

        # 懒初始化融合 MLP（输入维度 = C_seg (+ C_txt) + in_features 的门控输入）
        if self.fuse_mlp is None and self.seg_feat is not None:
            c_seg = self.seg_feat.size(1)
            c_txt = (self.text_feat.size(1) if self.text_feat is not None else 0)
            fuse_in = c_seg + c_txt + self.base.in_features  # 把 x 的均值门控也并进来更稳
            hidden = max(256, self.base.in_features // 2)

            self.fuse_mlp = nn.Sequential(
                nn.Linear(fuse_in, hidden, bias=False),
                nn.SiLU(),
                nn.Linear(hidden, self.base.in_features, bias=False),
            ).to(device=dev, dtype=dty)

    def forward(self, x):
        """
        x: [B, N, in_features]（注意 CrossAttention 前后的 Linear 都是 3D）
        """
        out = self.base(x)

        # 仅首层打印一次实际形状
        if (self.seg_feat is not None) and (not self._shape_printed):
            tf_shape = None if self.text_feat is None else tuple(self.text_feat.shape)
            print(f"[LoRALinear.forward] x={tuple(x.shape)}, seg_feat={tuple(self.seg_feat.shape)}, text_feat={tf_shape}")
            self._shape_printed = True

        if self.seg_feat is not None and self.fuse_mlp is not None:
            x_mean = x.mean(dim=1)  # [Bx, in_f]
            Bx = x_mean.shape[0]
            Bseg = self.seg_feat.shape[0]

            seg_feat = self.seg_feat
            text_feat = self.text_feat

            # 更稳健的批量对齐
            if Bseg != Bx:
                if Bseg == 0:
                    # 防御性处理
                    print("[LoRALinear] Warning: seg_feat is empty, skipping modulation.")
                    x_mod = x
                else:
                    repeat_factor = max(1, math.ceil(Bx / Bseg))
                    seg_feat = seg_feat.repeat_interleave(repeat_factor, dim=0)[:Bx]
                    if text_feat is not None:
                        text_feat = text_feat.repeat_interleave(repeat_factor, dim=0)[:Bx]

                    if not hasattr(self, "_print_once"):
                        print(f"[LoRALinear] batch mismatch: seg_feat {Bseg}→{Bx}, repeat_factor={repeat_factor}")
                        self._print_once = True
            else:
                repeat_factor = 1

            # === 拼接融合 ===
            if text_feat is not None:
                fuse_in = torch.cat([x_mean, seg_feat, text_feat], dim=-1)
            else:
                fuse_in = torch.cat([x_mean, seg_feat], dim=-1)

            delta = self.fuse_mlp(fuse_in)  # [B, in_f]
            delta = delta.unsqueeze(1).expand_as(x)
            x_mod = x + delta
        else:
            x_mod = x

        return out + self.B(self.A(x_mod)) * self.scaling


def lora_qkv_combine(module: nn.Module, r=16):
    """
    在给定 module（UNet 或 ControlNet 任意子树）里，把所有 Cross-Attn 的
    to_q / to_k / to_v 线性层替换为 LoRALinear（QKV 全覆盖）。
    """
    n = 0
    for m in module.modules():
        # 这三个属性名在 ldm 里通常就是 to_q/to_k/to_v（线性层）
        if hasattr(m, "to_q") and isinstance(m.to_q, nn.Linear):
            m.to_q = LoRALinear(m.to_q, r=r); n += 1
        if hasattr(m, "to_k") and isinstance(m.to_k, nn.Linear):
            m.to_k = LoRALinear(m.to_k, r=r); n += 1
        if hasattr(m, "to_v") and isinstance(m.to_v, nn.Linear):
            m.to_v = LoRALinear(m.to_v, r=r); n += 1
    print(f"[inject] LoRA-wrapped Q/K/V linears = {n}")
    return n


@torch.no_grad()
def set_lora_condition(root: nn.Module, seg_feat: torch.Tensor, text_tokens: torch.Tensor):
    """
    仅给 LoRALinear 实例注入条件；不会碰到 MemoryEfficientCrossAttention。
    """
    for m in root.modules():
        if isinstance(m, LoRALinear):
            m.set_control_feature(seg_feat, text_tokens)


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


# 1. build model and load ckpt
cfg = OmegaConf.load(CFG)
model = instantiate_from_config(cfg.model)
sd = torch.load(CKPT, map_location="cpu"); sd = sd.get("state_dict", sd)
missing, unexpected = model.load_state_dict(sd, strict=False)
print("loaded ckpt. missing:", len(missing), "unexpected:", len(unexpected))

# move to GPU FIRST
model = model.to(device).eval()

# frozen: VAE / UNet
for p in model.first_stage_model.parameters(): p.requires_grad=False
for p in model.model.parameters(): p.requires_grad=False

# then add LoRA so A/B are created on the same device
wrapped = lora_qkv_combine(model.model.diffusion_model, r=LORA_R)
print(f"[inject] LoRA-wrapped attention layers (Q/K/V): {wrapped}")

for p in model.first_stage_model.parameters():
    p.requires_grad = False
for p in model.model.parameters():
    p.requires_grad = False

# LoRA 参数设为可训练
for m in model.model.diffusion_model.modules():
    if isinstance(getattr(m, "to_q", None), LoRALinear):
        for p in m.to_q.parameters(): p.requires_grad = True
    if isinstance(getattr(m, "to_k", None), LoRALinear):
        for p in m.to_k.parameters(): p.requires_grad = True
    if isinstance(getattr(m, "to_v", None), LoRALinear):
        for p in m.to_v.parameters(): p.requires_grad = True

# disable_unet_checkpointing(model.model.diffusion_model)
# print("UNet gradient checkpointing disabled.")

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

disable_bert_dropout(textenc.hf_bert)
print("[debug] BERT dropout disabled.")


adapter_params = [p for p in textenc.adapter.parameters() if p.requires_grad]
hf_params      = [p for p in textenc.hf_bert.parameters() if p.requires_grad]  # 当前应为空，因为冻结了
lora_params    = [p for n,p in model.model.diffusion_model.named_parameters()
                  if p.requires_grad and (".A.weight" in n or ".B.weight" in n)]

print(f"[check] adapter params: {sum(p.numel() for p in adapter_params)}")
print(f"[check] hf_bert params: {sum(p.numel() for p in hf_params)}")
print(f"[check] lora params:    {sum(p.numel() for p in lora_params)}")

# 至少要有一组非空
assert (len(adapter_params) + len(hf_params) + len(lora_params)) > 0, "No trainable params collected!"

# === 构建优化器 param_groups ===
param_groups = [
    {"params": adapter_params + hf_params, "lr": 7e-5, "weight_decay": 0.01},
    {"params": lora_params,               "lr": 1e-4, "weight_decay": 0.001},
]

opt = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))

use_cuda = torch.cuda.is_available()
scaler = GradScaler(enabled=use_cuda)

# lr dispatch: warmup + cosine
WARMUP = 2000
TOTAL_STEPS = STEPS
def set_warmup_lr(step):
    scale = max(0.1, min(1.0, step / float(WARMUP)))  # 前 10% 起步，线性到 1.0
    # 基准 lr 与上面 param_groups 保持一致
    base_lrs = [7e-5, 1e-4]
    for g, base in zip(opt.param_groups, base_lrs):
        g["lr"] = base * scale

from math import pi, cos
def cosine_decay_lr(step):
    t = (step - WARMUP) / max(1, (TOTAL_STEPS - WARMUP))
    base_lrs = [7e-5, 1e-4]
    LR_FLOOR = 2e-5
    for g, base in zip(opt.param_groups, base_lrs):
        g["lr"] = max(LR_FLOOR, base * 0.5 * (1 + cos(pi * min(1.0, t))))

# 2. data split and build DataLoader
os.makedirs(SPLIT_DIR, exist_ok=True)

# 固定数据对
ds_base = PairsJSONLDataset(JSONL, ROOTDIR, image_size=SIZE, seg_mode="RGB")
ds_val = Subset(ds_base, [0, 9, 18, 27, 36, 45, 54, 63])

# sanity check
print("[base] len:", len(ds_base))  # 应该是 1256

# 3. 用缓存后的 ds_train 来建 DataLoader
dl_val = DataLoader(ds_val,   batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

group_size = 8
num_samples = len(ds_base)
num_groups = num_samples // group_size

# === 1. same-seg 分组（每 8 条一组，不足补齐）
seg_indices = []
for i in range(0, num_samples, group_size):
    group = list(range(i, min(i + group_size, num_samples)))
    # 若最后不足 8 张，就循环补齐
    while len(group) < group_size:
        group.append(group[-1])  # 重复最后一张
    seg_indices.append(group)

# === 2. same-prompt 分组（第 0 条、第 8 条、第 16 条... 属于同一个 prompt）
prompt_indices = []
for offset in range(group_size):
    group = list(range(offset, num_samples, group_size))
    # 同样补齐到 8 张
    while len(group) < group_size:
        group.append(group[-1])
    prompt_indices.append(group)

# === 3. 交替混合
mixed_indices = []
num_seg_groups = len(seg_indices)
num_prompt_groups = len(prompt_indices)
prompt_group_len = len(prompt_indices[0])

for i in range(num_seg_groups):
    mixed_indices.extend(seg_indices[i])
    prompt_id = i % num_prompt_groups
    start = (i // num_prompt_groups) * group_size
    prompt_group = prompt_indices[prompt_id][start:start + group_size]
    while len(prompt_group) < group_size:
        prompt_group.append(prompt_indices[prompt_id][-1])
    mixed_indices.extend(prompt_group)

# === 4. 构造最终 dataset 和 dataloader
ds_mixed = Subset(ds_base, mixed_indices)
dl = DataLoader(ds_mixed, batch_size=group_size, shuffle=False, num_workers=0, pin_memory=False)

print(f"[mixed] total len = {len(ds_mixed)}  (≈ {len(mixed_indices)} samples)")
print(f"[mixed] first 40 indices: {mixed_indices[:40]}")

print("[check] train len:", len(ds_base))
print("[check] val len:",   len(ds_val))
vb = next(iter(dl_val))
print("=== sanity check dl_val batch ===")
for i in range(len(vb["prompt"])):
    seg_path = vb["seg_path"][i]
    gt_path  = vb["gt_path"][i]
    seg_sum  = float(vb["seg"][i].sum())
    print(f"[{i}] prompt={vb['prompt'][i]!r}")
    print(f"     seg_path={seg_path}")
    print(f"     gt_path={gt_path}")
    print(f"     seg_sum={seg_sum:.2f}")
print("==============================")

random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
train_meter = MeterEMA(beta=0.98)
best_val = float("inf")
best_score = float("inf")
BEST_SELECTOR = "l1"
ALPHA = 0.5

loss_log = _load_loss_log(LOSS_LOG_PATH)

trainable_lora = [n for n,p in model.model.diffusion_model.named_parameters()
                  if (".A.weight" in n or ".B.weight" in n) and p.requires_grad]
print(f"[check] trainable LoRA params: {len(trainable_lora)}")
pooler_params = [n for n,_ in textenc.hf_bert.pooler.named_parameters()] if hasattr(textenc.hf_bert,"pooler") else []
print("[bert] any pooler requires_grad?:",
      any(p.requires_grad for p in textenc.hf_bert.pooler.parameters()) if hasattr(textenc.hf_bert,"pooler") else False)


def build_hf_inputs(prompts, tokenizer, device, max_len):
    toks = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    # 移到正确设备
    toks = {k: v.to(device) for k, v in toks.items()}
    return toks

# 3. training loop
global_step = 0
while global_step < STEPS:
    for batch in dl:
        global_step += 1
        seg = batch["seg"].to(device=device, dtype=torch.float32)
        gt  = batch["gt"].to(device=device, dtype=torch.float32)
        prompts = batch["prompt"]

        with torch.no_grad():
            z = model.get_first_stage_encoding(model.encode_first_stage(gt)).detach()  # [B,4,H/8,W/8]

        # z = z.to(torch.float16)

        t = torch.randint(0, model.num_timesteps, (z.size(0),), device=device).long()
        noise  = torch.randn_like(z)
        z_noisy= model.q_sample(z, t, noise=noise)

        # z_noisy.requires_grad_(True)

        with autocast(device_type='cuda', enabled=(device=="cuda"), dtype=torch.float16):
            # text context (complete tokens)
            c_cross = textenc.encode(prompts)                 # [B,L,768], fp32 on GPU
            c_cross = c_cross.to(device=device, dtype=torch.float16)
            # drop p=0.2 text prompt
            if model.training and torch.rand(1, device=device).item() < TEXT_DROP_PROB:
                c_cross = torch.zeros_like(c_cross)

            if global_step == 1:
                print("c_cross requires_grad?", c_cross.requires_grad)

            # seg: [B, 3, H, W]
            with torch.no_grad():
                B = seg.size(0)
                emb = torch.zeros(B, model.control_model.model_channels * 4, device=device, dtype=seg.dtype)
                control_states = model.control_model.input_hint_block(seg, emb, context=None)
                seg_feat = F.adaptive_avg_pool2d(control_states, 1).flatten(1)

            text_tokens = c_cross
            set_lora_condition(model.model.diffusion_model, seg_feat, text_tokens)

            cond = {"c_crossattn": [c_cross], "c_concat": [seg]}
            eps_hat = model.apply_model(z_noisy, t, cond)

            loss = F.mse_loss(eps_hat.float(), noise.float())

            ema  = train_meter.update(loss.item())
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

        if global_step == 1:
            if isinstance(control_states, (tuple, list)):
                for i, s in enumerate(control_states):
                    print(f"[ControlEncoder] layer{i}: shape={tuple(s.shape)}, dtype={s.dtype}")
                seg_feat = F.adaptive_avg_pool2d(control_states[-1], 1).flatten(1)
            else:
                print(f"[ControlEncoder] output: shape={tuple(control_states.shape)}, dtype={control_states.dtype}")
                seg_feat = F.adaptive_avg_pool2d(control_states, 1).flatten(1)
            print(f"[Fuse seg_feat shape] {tuple(seg_feat.shape)}, dtype={seg_feat.dtype}")

        if global_step % 200 == 0:
            print(f"[{global_step}/{STEPS}] train_mse={loss.item():.4f}  ema={ema:.4f}")
            torch.cuda.empty_cache()

        if global_step % VAL_EVERY == 0:
            model.eval(); textenc.eval()
            val_batch = next(iter(dl_val))
            val_mse = validate_step(model, textenc, val_batch, device, max_length=MAXLEN,
                                    iters=8, fixed_t=None, fixed_noise=None)
            print(f"[val] mse={val_mse:.4f}")
            model.train(); textenc.train()

            # visualization: sampling some output combined figures (seg, pred, gt, prompt)
            if global_step % SAVE_IMG_EVERY == 0:
                model.eval(); textenc.eval()
                with torch.no_grad():
                    # 拿到整套 10 条（batch=10）
                    for vb in dl_val:
                        seg_v = vb["seg"].to(device, torch.float32)
                        gts_v = vb["gt"].to(device, torch.float32)  # [10,C,H,W]
                        proms = vb["prompt"]  # List[str], 长度 10
                        K = gts_v.size(0)

                        # 逐条跑采样（更省显存；如果够用也可一次性循环拼 batch=1）
                        segs_list, preds_list, gts_list = [], [], []
                        for i in range(K):
                            seg_i = seg_v[i:i + 1]
                            gt_i = gts_v[i:i + 1]
                            p_i = [proms[i]]
                            pred_i = sample_preview_CNLora(
                                model, textenc,
                                {"seg": seg_i, "gt": gt_i, "prompt": p_i},
                                device, steps=12, scale=7.5, eta=0.0, seed=1234
                            )

                            segs_list.append(seg_i)
                            preds_list.append(pred_i)
                            gts_list.append(gt_i)

                        out_png = os.path.join(OUTDIR, f"vis/concept_{global_step:06d}.png")
                        save_concept_grid_new(segs_list, preds_list, gts_list, proms, out_png)
                        break  # 只可视化一组（10条）
                model.train(); textenc.train()

            _save_loss_log(LOSS_LOG_PATH, loss_log)
            _save_loss_plot(loss_log, os.path.join(OUTDIR, "vis", "loss_curve.png"))

            torch.save(textenc.state_dict(), os.path.join(OUTDIR, f"textenc_adapter_step{global_step:06d}.pt"))
            lora_sd = {
                k: v for k, v in model.model.diffusion_model.state_dict().items()
                if any(s in k for s in [".to_q.A.weight", ".to_q.B.weight",
                                        ".to_k.A.weight", ".to_k.B.weight",
                                        ".to_v.A.weight", ".to_v.B.weight"])
            }
            torch.save(lora_sd, os.path.join(OUTDIR, f"unet_lora_step{global_step:06d}.pt"))
            print(f"   -> saved recent pt at step {global_step}")

            model.train(); textenc.train()

    if global_step >= STEPS:
        break

# save current ckpt at the end
torch.save(textenc.state_dict(), os.path.join(OUTDIR, "last_text_lastlayer_adapter.pt"))
lora_sd = {
    k: v for k, v in model.model.diffusion_model.state_dict().items()
    if any(s in k for s in [".to_q.A.weight", ".to_q.B.weight",
                            ".to_k.A.weight", ".to_k.B.weight",
                            ".to_v.A.weight", ".to_v.B.weight"])
}
torch.save(lora_sd, os.path.join(OUTDIR, "last_unet_lora_kv.pt"))
print("done.")

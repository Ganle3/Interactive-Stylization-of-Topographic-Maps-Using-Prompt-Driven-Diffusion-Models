# -*- coding: utf-8 -*-
"""
Training: fine-tune last 2 layers of CLIP text encoder for SD1.5 + ControlNet
- Keep your data split & DataLoader from data_utils (PairsJSONLDataset)
- Freeze VAE/UNet/ControlNet; only train FrozenCLIPEmbedder last N layers
- Auto-prefix each prompt with "map in swiss topo style, "
- Loss logging + simple validation + periodic sampling/preview
"""

import os, math, json, sys
sys.path.append(r"D:\Junyhuang\Project2\ControlNet")

import matplotlib.pyplot as plt

# ---- Hard-disable gradient checkpointing (as in your original script) ----
def _no_checkpoint(func, inputs, params=None, flag=None):
    return func(*inputs)

import importlib
ldm_util = importlib.import_module("ldm.modules.diffusionmodules.util")
ldm_util.checkpoint = _no_checkpoint

ldm_attn = importlib.import_module("ldm.modules.attention")
ldm_attn.checkpoint = _no_checkpoint

print("[patch] Disabled ldm checkpoint: util.checkpoint AND attention.checkpoint are now no-ops.")

# ---- Torch / SD imports ----
import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from ControlNet.ldm.util import instantiate_from_config
from ControlNet.ldm.models.diffusion.ddim import DDIMSampler
from torch.amp import GradScaler, autocast

# utilities (KEEP): split/dataset/metrics/vis
from data_utils import stratified_split_ratio, PairsJSONLDataset
from vis_metrics import MeterEMA  # 保留EMA用


# ===================== Config =====================
CKPT    = r"D:\Junyhuang\Project2\BaseModel\Swisstopo.ckpt"
CFG     = r"D:\Junyhuang\Project2\ControlNet\models\cldm_v15.yaml"
ROOTDIR = r"D:\Junyhuang\Project2_Data\Training Data\Item_color"
JSONL   = os.path.join(ROOTDIR, "meta", "pairs.jsonl")
SPLIT_DIR = os.path.join(ROOTDIR, "meta_splits")
OUTDIR  = r"D:\Junyhuang\Project2\Outputs\clip_last2layers_color"
LOSS_LOG_PATH = os.path.join(OUTDIR, "vis", "loss_log.json")
os.makedirs(os.path.join(OUTDIR, "vis"), exist_ok=True)

# Training hyperparameters
BATCH = 8
SIZE = 512
STEPS = 3000
MAXLEN = 77
VAL_EVERY = 100
SAVE_IMG_EVERY = 100
device = "cuda" if torch.cuda.is_available() else "cpu"

# Text settings
STYLE_PREFIX = "map in swiss topo style, "
TRAIN_LAST_N_LAYERS = 2       # only train CLIP encoder last N layer
LR_TEXT = 1e-5
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.999)

# Sampler settings
SAMPLE_STEPS = 30
GUIDANCE_SCALE = 7.5
SEED = 1234

# Loss logging / plotting
def _load_loss_log(path):
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                for k in ["step", "train_mse", "ema", "val_step", "val_mse", "train_l1"]:
                    data.setdefault(k, [])
                return data
            except Exception:
                pass
    return {"step": [], "train_mse": [], "ema": [], "val_step": [], "val_mse": [], "train_l1": []}

def _save_loss_log(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)

def _save_loss_plot(data, out_png):
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

# Build model & freeze/unfreeze
cfg = OmegaConf.load(CFG)
model = instantiate_from_config(cfg.model).to(device)

sd = torch.load(CKPT, map_location="cpu")
sd = sd.get("state_dict", sd)
missing, unexpected = model.load_state_dict(sd, strict=False)
print("loaded ckpt. missing:", len(missing), "unexpected:", len(unexpected))

from ControlNet.ldm.modules.encoders.modules import FrozenCLIPEmbedder

if isinstance(model.cond_stage_model, FrozenCLIPEmbedder):
    te = model.cond_stage_model
    te.train_last_n_layers = TRAIN_LAST_N_LAYERS
    te.freeze()
    print(f"[TextEnc] Using CKPT text encoder; train_last_n_layers={TRAIN_LAST_N_LAYERS}")
else:
    old_te = model.cond_stage_model
    te = FrozenCLIPEmbedder(
        version="openai/clip-vit-large-patch14",
        device=device, max_length=MAXLEN, freeze=True,
        layer="last", train_last_n_layers=TRAIN_LAST_N_LAYERS
    ).to(device)
    try:
        te.load_state_dict(old_te.state_dict(), strict=False)
        print("[TextEnc] Loaded weights from CKPT cond_stage_model into new FrozenCLIPEmbedder.")
    except Exception as e:
        print("[TextEnc] WARNING: could not load weights from old cond_stage_model:", e)
    model.cond_stage_model = te

# freeze VAE / U-Net / ControlNet
for p in model.first_stage_model.parameters():
    p.requires_grad = False

if hasattr(model, "model"):  # U-Net
    for p in model.model.parameters():
        p.requires_grad = False

if hasattr(model, "control_model"):
    for p in model.control_model.parameters():
        p.requires_grad = False

# only fine-tuning TextEncoder trainable parameters (unfreeze last layers)
te_params = [p for p in model.cond_stage_model.parameters() if p.requires_grad]
assert len(te_params) > 0, "No trainable params in TextEncoder. Check TRAIN_LAST_N_LAYERS or freeze()."
opt = torch.optim.AdamW(te_params, lr=LR_TEXT, betas=BETAS, weight_decay=WEIGHT_DECAY)
scaler = GradScaler(enabled=(device == "cuda"))

# close U-Net checkpointing / open xformers
def disable_unet_checkpointing(unet):
    if hasattr(unet, "use_checkpoint"):
        unet.use_checkpoint = False
    for m in unet.modules():
        if hasattr(m, "checkpoint"):
            m.checkpoint = False

if hasattr(model, "model") and hasattr(model.model, "diffusion_model"):
    disable_unet_checkpointing(model.model.diffusion_model)
    print("UNet gradient checkpointing disabled.")
try:
    model.model.diffusion_model.set_use_memory_efficient_attention_xformers(True)
    print("Enabled xformers memory efficient attention.")
except Exception as e:
    print("xformers not available:", e)


# Data split & loaders
os.makedirs(SPLIT_DIR, exist_ok=True)
split_paths = stratified_split_ratio(
    JSONL, SPLIT_DIR, per_prompt_ratio=(0.8, 0.1, 0.1),
    min_per_split=(1, 1, 1), shuffle_seed=42, verbose=True
)
train_jsonl = split_paths["train"]
val_jsonl   = split_paths["val"]
test_jsonl  = split_paths["test"]

dl     = DataLoader(PairsJSONLDataset(train_jsonl, ROOTDIR, image_size=SIZE, seg_mode="RGB"),
                    batch_size=BATCH, shuffle=True,  num_workers=0, pin_memory=True)
dl_val = DataLoader(PairsJSONLDataset(val_jsonl,   ROOTDIR, image_size=SIZE, seg_mode="RGB"),
                    batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)

torch.backends.cudnn.benchmark = True
train_meter = MeterEMA(beta=0.98)
loss_log = _load_loss_log(LOSS_LOG_PATH)


# Helper: prompt prefix + sampler + simple metrics
def add_style_prefix(texts):
    pref = STYLE_PREFIX.lower()
    out = []
    for t in texts:
        s = (t or "").strip()
        if not s.lower().startswith(pref):
            s = STYLE_PREFIX + s
        out.append(s)
    return out

@torch.no_grad()
def sample_preview_local(model, seg, prompts, steps=SAMPLE_STEPS, scale=GUIDANCE_SCALE, seed=SEED):
    sampler = DDIMSampler(model)
    torch.manual_seed(seed)
    seg = seg.to(device)
    prompts = add_style_prefix(prompts)

    cond = {
        "c_concat": [seg],
        "c_crossattn": [model.get_learned_conditioning(prompts)]
    }
    un_cond = {
        "c_concat": [seg],
        "c_crossattn": [model.get_learned_conditioning([""] * len(prompts))]
    }

    # latent shape (C=4, D=8)
    H, W = seg.shape[-2:]
    shape = (4, H // 8, W // 8)
    samples, _ = sampler.sample(
        S=steps, conditioning=cond, batch_size=seg.size(0), shape=shape,
        verbose=False, unconditional_guidance_scale=scale, unconditional_conditioning=un_cond, eta=0.0
    )
    x = model.decode_first_stage(samples)
    x = (x.clamp(-1, 1) + 1.0) / 2.0  # [0,1]
    return x

def simple_l1(pred_01, gt_01):
    # pred_01, gt_01: [B,3,H,W], range [0,1]
    return torch.mean(torch.abs(pred_01 - gt_01)).item()

def save_grid(imgs_bchw, path, nrow=2):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    from torchvision.utils import save_image, make_grid
    grid = make_grid(imgs_bchw, nrow=nrow)
    save_image(grid, path)


# Training loop (matches original noise training style)
# check before training
te = model.cond_stage_model
total = sum(p.numel() for p in te.parameters())
trainable = [(n,p.numel()) for n,p in te.named_parameters() if p.requires_grad]
print(f"[SANITY] TextEncoder total params: {total:,}")
print(f"[SANITY] TextEncoder trainable params: {sum(n for _,n in trainable):,}")
print("[SANITY] Trainable list (show first 10):")
for i,(n,_) in enumerate(trainable[:10]):
    print("  ", n)

assert len(trainable) > 0, "No trainable params in text encoder—check train_last_n_layers and freeze()!"

model.train()
global_step = 0
best_score = float("inf")

while global_step < STEPS:
    for batch in dl:
        global_step += 1

        seg = batch["seg"].to(device=device, dtype=torch.float32)      # ControlNet hint [0,1]
        gt  = batch["gt"].to(device=device, dtype=torch.float32)       # target RGB [-1,1] expected by first-stage
        prompts = batch["prompt"]                                      # list[str]

        # encode to latents
        with torch.no_grad():
            z = model.get_first_stage_encoding(model.encode_first_stage(gt))  # [B,4,H/8,W/8]

        # diffusion training target
        t = torch.randint(0, model.num_timesteps, (z.size(0),), device=device).long()
        noise  = torch.randn_like(z)
        z_noisy= model.q_sample(z, t, noise=noise)

        # build cond with prefixed prompts
        prompts_pref = add_style_prefix(prompts)
        with autocast(device_type='cuda', enabled=(device == "cuda"), dtype=torch.float16):
            cond = {"c_crossattn": [model.get_learned_conditioning(prompts_pref)],
                    "c_concat":    [seg]}
            eps_hat = model.apply_model(z_noisy, t, cond)
            loss = F.mse_loss(eps_hat.float(), noise.float())

        ema = train_meter.update(loss.item())
        loss_log["step"].append(int(global_step))
        loss_log["train_mse"].append(float(loss.item()))
        loss_log["ema"].append(float(ema))

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(te_params, max_norm=1.0)
        scaler.step(opt)
        scaler.update()

        if global_step % 100 == 0:
            print(f"[{global_step}/{STEPS}] train_mse={loss.item():.4f}  ema={ema:.4f}")

        # validation
        if (global_step % VAL_EVERY) == 0:
            model.eval()
            with torch.no_grad():
                # 1) MSE on a few val batches
                val_iter = iter(dl_val)
                n_batches = min(8, len(dl_val))
                val_sum = 0.0; n_count = 0
                for _ in range(n_batches):
                    try:
                        vb = next(val_iter)
                    except StopIteration:
                        break
                    seg_v = vb["seg"].to(device=device, dtype=torch.float32)
                    gt_v  = vb["gt"].to(device=device, dtype=torch.float32)
                    prompts_v = add_style_prefix(vb["prompt"])

                    z_v = model.get_first_stage_encoding(model.encode_first_stage(gt_v))
                    t_v = torch.randint(0, model.num_timesteps, (z_v.size(0),), device=device).long()
                    n_v = torch.randn_like(z_v)
                    zq  = model.q_sample(z_v, t_v, noise=n_v)
                    cond_v = {"c_crossattn":[model.get_learned_conditioning(prompts_v)],
                              "c_concat":[seg_v]}
                    eps_hat_v = model.apply_model(zq, t_v, cond_v)
                    val_mse = F.mse_loss(eps_hat_v.float(), n_v.float()).item()
                    val_sum += val_mse; n_count += 1
                val_mse_avg = val_sum / max(1, n_count)

                # 2) Preview & L1
                take = min(4, seg.size(0))  # seg/gt/prompts on training data
                seg0 = seg[:take].detach()
                gt0 = gt[:take].detach()
                prom0 = prompts[:take]
                prom_pref = add_style_prefix(prom0)
                prom_empty = [STYLE_PREFIX.strip()] * take
                pred_with = sample_preview_local(model, seg0, prom_pref, steps=SAMPLE_STEPS, scale=GUIDANCE_SCALE, seed=SEED)
                pred_empty = sample_preview_local(model, seg0, prom_empty, steps=SAMPLE_STEPS, scale=GUIDANCE_SCALE, seed=SEED)

                gt0_01 = (gt0.clamp(-1,1)+1)/2
                train_l1 = simple_l1(pred_with, gt0_01)

                print(f"   -> val_mse={val_mse_avg:.4f}  train_l1={train_l1:.4f}")

                loss_log["val_step"].append(int(global_step))
                loss_log["val_mse"].append(float(val_mse_avg))
                loss_log["train_l1"].append(float(train_l1))

                # save preview
                if global_step % SAVE_IMG_EVERY == 0:
                    rows = []
                    for i in range(take):
                        row = torch.cat([
                            seg0[i].unsqueeze(0),
                            gt0_01[i].unsqueeze(0),
                            pred_with[i].unsqueeze(0),
                            pred_empty[i].unsqueeze(0)
                        ], dim=0)
                        rows.append(row)

                    grid = torch.cat(rows, dim=0)
                    grid = make_grid(grid, nrow=4)
                    os.makedirs(os.path.join(OUTDIR, "vis"), exist_ok=True)
                    out_png = os.path.join(OUTDIR, f"vis/train_{global_step:06d}.png")
                    save_image(grid, out_png)
                    print(f"   -> saved TRAIN preview grid to {out_png}")
                    print("   -> prompts:")
                    for p in prom_pref:
                        print("      ", p)

                _save_loss_log(LOSS_LOG_PATH, loss_log)
                _save_loss_plot(loss_log, os.path.join(OUTDIR, "vis", "loss_curve.png"))

                # best on L1
                if train_l1 < best_score:
                    best_score = train_l1
                    os.makedirs(os.path.join(OUTDIR, "ckpt"), exist_ok=True)
                    torch.save({"state_dict": model.state_dict()},
                               os.path.join(OUTDIR, "ckpt", f"best_step{global_step}.ckpt"))
                    print(f"   -> saved BEST (L1={best_score:.4f}) at step {global_step}")

            model.train()

        if global_step >= STEPS:
            break

# Final save
os.makedirs(os.path.join(OUTDIR, "ckpt"), exist_ok=True)
torch.save({"state_dict": model.state_dict()},
           os.path.join(OUTDIR, "ckpt", f"last_step{global_step}.ckpt"))
print("done.")

import os, math, json, sys
sys.path.append(r"D:\Junyhuang\Project2\ControlNet")
import matplotlib.pyplot as plt

import os, json, torch, numpy as np
from PIL import Image
from omegaconf import OmegaConf
from ControlNet.ldm.util import instantiate_from_config
from transformers import AutoTokenizer, AutoModel
from torchvision.utils import make_grid, save_image
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF
import contextlib
from torch.amp import autocast
import types
import torch.nn as nn

# =====================================================
# === 写死路径（请根据你两个实验目录改一下这四行） ===
# =====================================================

CFG = r"D:\Junyhuang\Project2\ControlNet\models\cldm_v15.yaml"
CKPT = r"D:\Junyhuang\Project2\BaseModel\Swisstopo.ckpt"
TRAIN_JSONL = r"D:\Junyhuang\Project2_Data\Training Data\Item_color\meta_splits\pairs_val.jsonl"
DATA_ROOT = r"D:\Junyhuang\Project2_Data\Training Data\Item_color"

# RunA: 第一次训练的权重
TEXT_PT_A = r"D:\Junyhuang\Project2\Outputs\sdfusion_bert_lora_color_org\best_text_lastlayer.pt"
LORA_PT_A = r"D:\Junyhuang\Project2\Outputs\sdfusion_bert_lora_color_org\best_unet_lora_kv.pt"

# RunB: 第二次训练的权重
TEXT_PT_B = r"D:\Junyhuang\Project2\Outputs\sdfusion_bert_lora_color_org\last_text_lastlayer.pt"
LORA_PT_B = r"D:\Junyhuang\Project2\Outputs\sdfusion_bert_lora_color_org\last_unet_lora_kv.pt"

OUTDIR = r"D:\Junyhuang\Project2\Outputs\compare_model"

# 采样与批量
INDEXES = [0, 1]     # 从 train 里取这几条组成一个小批次（可改）
SIZE    = 512
STEPS   = 30
SCALE   = 7.5
ETA     = 0.0
SEED    = 1234

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ================== 小工具函数 ==================
def _install_lora_on_linear(lin: nn.Linear, r: int, scaling: float = 1.0):
    if getattr(lin, "__lora_installed__", False):
        return

    # 目标设备/精度与原 Linear 对齐
    dev   = lin.weight.device
    dtype = lin.weight.dtype

    # 用子模块，保证 state_dict 键是 ".A.weight"/".B.weight"
    lin.A = nn.Linear(lin.in_features, r, bias=False)
    lin.B = nn.Linear(r, lin.out_features, bias=False)
    nn.init.zeros_(lin.A.weight)
    nn.init.zeros_(lin.B.weight)

    # 关键：把 A/B 移到与 lin 一致的 device / dtype
    lin.A.to(device=dev, dtype=dtype)
    lin.B.to(device=dev, dtype=dtype)

    lin.scaling = float(scaling)
    lin.__lora_installed__ = True

    _orig_forward = lin.forward

    def _lora_forward(self, x):
        y = _orig_forward(x)
        # 与 y 对齐 dtype；A/B 已经在同一 device
        x_cast = x.to(dtype=y.dtype)
        lora = self.B(self.A(x_cast))
        return y + lora * self.scaling

    lin.forward = types.MethodType(_lora_forward, lin)

def install_lora_kv_on_unet(unet: nn.Module, r: int):
    """
    在 UNet 的注意力层上安装 K/V 的 LoRA 槽位。
    若你的训练是 Q/K/V/O 全挂，把 ('to_k','to_v') 改成 ('to_q','to_k','to_v')，
    并为 to_out[0] 也安装。
    """
    for m in unet.modules():
        # K/V
        for name in ("to_k", "to_v"):
            lin = getattr(m, name, None)
            if isinstance(lin, nn.Linear):
                _install_lora_on_linear(lin, r=r, scaling=1.0)
        # 如果你当时也给 to_out[0] 装了 LoRA，就取消下面的注释
        # if hasattr(m, "to_out") and hasattr(m.to_out, "__getitem__"):
        #     lin = m.to_out[0]
        #     if isinstance(lin, nn.Linear):
        #         _install_lora_on_linear(lin, r=r, scaling=1.0)

def infer_lora_rank_from_sd(sd_lora: dict) -> int:
    """从 ckpt 的 .A.weight / .B.weight 推断 rank"""
    for k, v in sd_lora.items():
        if k.endswith(".A.weight"):
            return v.shape[0]  # [r, in_features]
    for k, v in sd_lora.items():
        if k.endswith(".B.weight"):
            return v.shape[1]  # [out_features, r]
    raise RuntimeError("无法从 LoRA ckpt 推断 rank（未找到 .A.weight/.B.weight）")

def load_cfg(cfg_path):
    return OmegaConf.load(cfg_path)

def build_model(cfg_path, ckpt_path):
    cfg = load_cfg(cfg_path)
    model = instantiate_from_config(cfg.model)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print(f"[CKPT] missing={len(missing)} unexpected={len(unexpected)}")
    model.to(DEVICE)
    return model

class SimpleBERTTextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased").to(DEVICE)
    def encode(self, prompts):
        enc = self.tokenizer(prompts, padding="max_length", truncation=True, max_length=77, return_tensors="pt")
        for k in enc: enc[k] = enc[k].to(DEVICE)
        out = self.model(**enc, output_hidden_states=True)
        return out.last_hidden_state

def load_textenc_from_trained_sd(simple_te: SimpleBERTTextEncoder, sd_path: str):
    sd = torch.load(sd_path, map_location="cpu")
    # 把 'hf_bert.' 前缀改成 'model.'（如果本来就是 'model.' 则不变）
    mapped = {}
    for k, v in sd.items():
        if k.startswith("hf_bert."):
            mapped["model." + k[len("hf_bert."):]] = v
        else:
            mapped[k] = v
    missing, unexpected = simple_te.load_state_dict(mapped, strict=False)
    print(f"[TEXTENC] load mapped from {os.path.basename(sd_path)} -> missing={len(missing)} unexpected={len(unexpected)}")

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def load_img(path):
    img = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.NEAREST)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.tensor(arr)

def to_batch(entries):
    segs, gts, prompts = [], [], []
    for e in entries:
        segs.append(load_img(os.path.join(DATA_ROOT, e["source"])))
        gts.append(load_img(os.path.join(DATA_ROOT, e["target"])))
        prompts.append(e["prompt"])
    seg = torch.stack(segs, 0).to(DEVICE)
    gt  = torch.stack(gts,  0).to(DEVICE)
    return {"seg": seg, "gt": gt, "prompt": prompts}

def ensure_minus1to1(x):
    """
    把 [0,1] 的张量拉回 [-1,1]；若本来就有负数/>1，视为已是 [-1,1]，直接返回。
    """
    xmin, xmax = float(x.min()), float(x.max())
    if -0.01 <= xmin <= 0.01 and 0.99 <= xmax <= 1.01:
        return x * 2.0 - 1.0
    return x

# @torch.no_grad()
# def save_ab_grid_runs(seg_batch, pred_a, pred_b, gt_batch, out_png, prompts_a):
#     """
#     每个样本一行四列：SEG | PRED(A) | PRED(B) | GT，并在行上方写：A: prompt | B: RunB
#     约定：pred / gt 输入为 [-1,1]，内部映射到 [0,1] 再保存。
#     """
#     from PIL import ImageDraw, ImageFont
#
#     os.makedirs(os.path.dirname(out_png), exist_ok=True)
#     segs = seg_batch.detach().float().cpu().clamp(0,1)
#     gts  = (gt_batch.detach().float().cpu().clamp(0,1) * 2 - 1)  # 若gt本就是0..1，拉回[-1,1]
#     pa   = ensure_minus1to1(pred_a.detach().float().cpu())
#     pb   = ensure_minus1to1(pred_b.detach().float().cpu())
#
#     # 映射到0..1
#     gts = (gts + 1) / 2
#     pa  = (pa  + 1) / 2
#     pb  = (pb  + 1) / 2
#
#     B, C, H, W = segs.shape
#     tiles = []
#     for i in range(B):
#         tiles += [segs[i], pa[i], pb[i], gts[i]]
#
#     grid = make_grid(tiles, nrow=4, padding=2)
#     save_image(grid, out_png)
#
#     img = TF.to_pil_image(grid).convert("RGBA")
#     overlay = Image.new("RGBA", img.size, (0,0,0,0))
#     draw = ImageDraw.Draw(overlay)
#     try:
#         font = ImageFont.load_default()
#     except Exception:
#         font = None
#     pad = 2
#     row_top = pad
#     for i in range(B):
#         text = f"A: {prompts_a[i]}   |   B: RunB (same prompt)"
#         rect_h = 16
#         draw.rectangle([(0, max(0, row_top - rect_h - 1)), (img.width, row_top - 1)], fill=(0,0,0,128))
#         draw.text((4, row_top - rect_h), text, fill=(255,255,255,255), font=font)
#         row_top += H + pad
#     img = Image.alpha_composite(img, overlay).convert("RGB")
#     img.save(out_png)
#     print(f"[SAVE] {out_png}")
# @torch.no_grad()
# def save_ab_grid_runs(
#     seg_batch, pred_a, pred_b, gt_batch, out_png,
#     prompts_a,
#     b_caption="keep original color"   # 右侧标题可按需改，比如 "keep original colors"
# ):
#     """
#     保存四联图（每样本一行）：
#       SEG | PRED(A) | PRED(B) | GT
#     与 sample_preview 逻辑对齐：
#       - pred_* / gt 输入为 [-1, 1]  -> 内部映射到 [0, 1]
#       - seg 视为 [0, 1]             -> 仅 clamp
#     会在每行图片上方叠加文字：A: <prompt> | B: <b_caption>
#     """
#     os.makedirs(os.path.dirname(out_png), exist_ok=True)
#
#     # ---- 数值域处理 ----
#     segs  = seg_batch.detach().float().cpu().clamp(0, 1)             # [0, 1]
#     pa    = pred_a.detach().float().cpu().clamp(-1, 1)
#     pb    = pred_b.detach().float().cpu().clamp(-1, 1)
#     gts   = gt_batch.detach().float().cpu().clamp(-1, 1)
#
#     # [-1,1] -> [0,1]
#     pa = (pa + 1.0) / 2.0
#     pb = (pb + 1.0) / 2.0
#     gts = (gts + 1.0) / 2.0
#
#     # 保障范围
#     pa = pa.clamp(0, 1)
#     pb = pb.clamp(0, 1)
#     gts = gts.clamp(0, 1)
#
#     B, C, H, W = segs.shape
#
#     # ---- 拼接格子 ----
#     tiles = []
#     for i in range(B):
#         tiles += [segs[i], pa[i], pb[i], gts[i]]
#
#     grid = make_grid(tiles, nrow=4, padding=2)   # 每行 4 张
#     save_image(grid, out_png)                    # 先存一份纯图
#
#     # ---- 叠加每行文字（半透明黑条）----
#     img = TF.to_pil_image(grid).convert("RGBA")
#     overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
#     draw = ImageDraw.Draw(overlay)
#     try:
#         font = ImageFont.load_default()
#     except Exception:
#         font = None
#
#     pad = 2
#     cell_h = H + pad  # 单行高度（含下方 padding）
#     row_top = pad
#     for i in range(B):
#         text = f"A: {str(prompts_a[i])}   |   B: {b_caption}"
#         rect_h = 16
#         # 顶部条位置：行图像上边缘上方 rect_h 像素
#         y0 = max(0, row_top - rect_h - 1)
#         y1 = max(0, row_top - 1)
#         draw.rectangle([(0, y0), (img.width, y1)], fill=(0, 0, 0, 128))
#         draw.text((4, y0 + 1), text, fill=(255, 255, 255, 255), font=font)
#         row_top += cell_h
#
#     img = Image.alpha_composite(img, overlay).convert("RGB")
#     img.save(out_png)
#     print(f"[SAVE] {out_png}")

@torch.no_grad()
def save_ab_grid(seg_batch, pred_a, pred_b, gt_batch, out_png, prompts_a, pred_b0=None):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    # seg / gt: 都是 [0,1]，只需 clamp
    segs = seg_batch.detach().float().cpu().clamp(0, 1)
    gts  = gt_batch.detach().float().cpu().clamp(0, 1)

    # preds: [-1,1] -> [0,1]
    pa = (pred_a.detach().float().cpu().clamp(-1, 1) + 1) / 2
    pb = (pred_b.detach().float().cpu().clamp(-1, 1) + 1) / 2
    if pred_b0 is not None:
        pb0 = (pred_b0.detach().float().cpu().clamp(-1, 1) + 1) / 2

    B, C, H, W = segs.shape
    tiles, ncol = [], (5 if pred_b0 is not None else 4)
    for i in range(B):
        row = [segs[i], pa[i], pb[i]]
        if pred_b0 is not None:
            row.append(pb0[i])
        row.append(gts[i])
        tiles += row

    grid = make_grid(tiles, nrow=ncol, padding=2)
    save_image(grid, out_png)  # 不要再额外归一化

    # 叠文字说明
    img = TF.to_pil_image(grid)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    pad = 2
    row_top = pad
    for i in range(B):
        txt = f"A: {prompts_a[i]}   |   B: (no text, CFG=1.0)"
        if pred_b0 is not None:
            txt += "   |   B0: (no text, LoRA=off, CFG=1.0)"
        rect_h = 16
        draw.rectangle([(0, max(0, row_top - rect_h - 1)), (img.width, row_top - 1)], fill=(0, 0, 0, 127))
        draw.text((4, row_top - rect_h), txt, fill=(255, 255, 255), font=font)
        row_top += H + pad

    img.save(out_png)


def count_active_lora_layers(unet):
    cnt = 0
    for m in unet.modules():
        for name in ("to_q", "to_k", "to_v", "to_out"):
            lin = getattr(m, name, None)
            if hasattr(lin, "scaling"):
                cnt += 1
    return cnt

# ================== 主流程 ==================
def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 读取 train 样本
    pairs = read_jsonl(TRAIN_JSONL)
    entries = [pairs[i % len(pairs)] for i in INDEXES]
    batch = to_batch(entries)
    print(f"[DATA] B={len(entries)}  seg range=({batch['seg'].min().item():.3f},{batch['seg'].max().item():.3f})")
    print(f"[DATA] prompt[0]: {batch['prompt'][0]}")

    # 构建 base 模型 & 文本编码器
    model = build_model(CFG, CKPT)
    textenc = SimpleBERTTextEncoder()

    # 进入 eval +（若有）EMA；并把 control_scales 设成与旧预览一致（通常1.0）
    ema_ctx = getattr(model, "ema_scope", None)
    ema_ctx = ema_ctx() if callable(ema_ctx) else contextlib.nullcontext()

    with torch.no_grad(), ema_ctx:
        model.eval()
        textenc.eval()
        old_scales = getattr(model, "control_scales", None)
        if old_scales is not None:
            model.control_scales = [1.0] * len(old_scales)

        model.to(device)
        textenc.to(device)

        # ==== Run A ====
        print(f"[LOAD A] text={TEXT_PT_A}")
        print(f"[LOAD A] lora={LORA_PT_A}")
        # sd_textA = torch.load(TEXT_PT_A, map_location="cpu")
        sd_loraA = torch.load(LORA_PT_A, map_location="cpu")

        # 1) 安装 LoRA 插槽（用 ckpt 推断 r）
        rA = infer_lora_rank_from_sd(sd_loraA)
        install_lora_kv_on_unet(model.model.diffusion_model, r=rA)

        # 2) 再加载 LoRA 权重
        missingA, unexpectedA = model.model.diffusion_model.load_state_dict(sd_loraA, strict=False)
        print(f"[A] lora missing={len(missingA)} unexpected={len(unexpectedA)}")

        # 3) 加载 textenc 权重
        load_textenc_from_trained_sd(textenc, TEXT_PT_A)
        # 4) 统计 LoRA 状态
        print(f"[A] active LoRA layers: {count_active_lora_layers(model.model.diffusion_model)}")
        # model.model.diffusion_model.load_state_dict(sd_loraA, strict=False)

        # A 版：同一批数据，真实 prompt vs 空 prompt
        B = batch["seg"].size(0)
        batch_A_real = {"seg": batch["seg"], "gt": batch["gt"], "prompt": batch["prompt"]}
        batch_A_keep = {"seg": batch["seg"], "gt": batch["gt"], "prompt": ["keep original color"] * B}

        preds_A_real = sample_preview(model, textenc, batch_A_real, device, steps=STEPS, scale=SCALE, eta=ETA,
                                      seed=SEED)
        preds_A_empty = sample_preview(model, textenc, batch_A_keep, device, steps=STEPS, scale=SCALE, eta=ETA,
                                       seed=SEED)

        out_png_A = os.path.join(OUTDIR, "vis", "ab_runA.png")
        save_ab_grid(batch["seg"], preds_A_real, preds_A_empty, batch["gt"], out_png_A, prompts_a=batch["prompt"])

        # ==== Run B ====
        print(f"[LOAD B] text={TEXT_PT_B}")
        print(f"[LOAD B] lora={LORA_PT_B}")
        # sd_textB = torch.load(TEXT_PT_B, map_location="cpu")
        sd_loraB = torch.load(LORA_PT_B, map_location="cpu")
        # 1) 安装 LoRA 插槽（用 ckpt 推断 r）
        rB = infer_lora_rank_from_sd(sd_loraB)
        install_lora_kv_on_unet(model.model.diffusion_model, r=rB)

        # 2) 再加载 LoRA 权重
        missingB, unexpectedB = model.model.diffusion_model.load_state_dict(sd_loraB, strict=False)
        print(f"[A] lora missing={len(missingB)} unexpected={len(unexpectedB)}")

        # 3) 加载 textenc 权重
        load_textenc_from_trained_sd(textenc, TEXT_PT_B)
        # model.model.diffusion_model.load_state_dict(sd_loraB, strict=False)
        # 4) 统计 LoRA 状态
        print(f"[B] active LoRA layers: {count_active_lora_layers(model.model.diffusion_model)}")

        # B 版：同一批数据，真实 prompt vs 空 prompt
        batch_B_real = {"seg": batch["seg"], "gt": batch["gt"], "prompt": batch["prompt"]}
        batch_B_empty = {"seg": batch["seg"], "gt": batch["gt"], "prompt": [""] * B}

        preds_B_real = sample_preview(model, textenc, batch_B_real, device, steps=STEPS, scale=SCALE, eta=ETA,
                                      seed=SEED)
        preds_B_empty = sample_preview(model, textenc, batch_B_empty, device, steps=STEPS, scale=SCALE, eta=ETA,
                                       seed=SEED)

        out_png_B = os.path.join(OUTDIR, "vis", "ab_runB.png")
        save_ab_grid(batch["seg"], preds_B_real, preds_B_empty, batch["gt"], out_png_B, prompts_a=batch["prompt"])

    print("[DONE] compare A/B finished.")


@torch.no_grad()
def sample_preview(model, textenc, batch, device, steps=30, scale=7.5, eta=0.0, seed=1234):
    """
    用 DDIM 采样若干步生成预测图，返回 pred_imgs in [-1,1]
    - seg（ControlNet hint）用 fp32
    - 文本条件用 fp16
    - 整个采样过程放在 autocast(fp16) 下，避免 Half/Float 冲突
    """
    from ControlNet.ldm.models.diffusion.ddim import DDIMSampler

    model.eval()
    seg = batch["seg"].to(device=device, dtype=torch.float32)     # hint用fp32
    gt  = batch["gt"].to(device=device, dtype=torch.float32)      # 仅用来拿尺寸就行
    prompts = batch["prompt"]
    B, _, H, W = seg.shape

    # 文本条件
    model_dtype = next(model.parameters()).dtype
    c_cross  = textenc.encode(prompts).to(device=device, dtype=model_dtype)
    uc_cross = torch.zeros_like(c_cross)

    cond = {"c_crossattn": [c_cross],  "c_concat": [seg]}
    uc   = {"c_crossattn": [uc_cross], "c_concat": [seg]}

    sampler = DDIMSampler(model)
    torch.manual_seed(seed)

    z_shape = (4, H // 8, W // 8)

    # 把整个采样过程放进 autocast(fp16)
    with autocast('cuda', enabled=(device == 'cuda'), dtype=torch.float16):
        samples, _ = sampler.sample(
            S=steps,
            conditioning=cond,
            batch_size=B,
            shape=z_shape,
            verbose=False,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc,
            eta=eta,
            x_T=None,
        )

    # 解码用 fp32 更稳
    pred_imgs = model.decode_first_stage(samples.to(torch.float32))
    pred_imgs = torch.clamp(pred_imgs, -1.0, 1.0)
    return pred_imgs

def vis_sample_preview(model, textenc, batch, steps=30, scale=7.5, eta=0.0):
    """
    封装成旧可视化的推理设置：
    - eval + ema_scope（上层已进入）
    - control_scales = 1.0
    - cond/uc 构造与训练一致（c_concat + c_crossattn）
    - 解码后返回 [-1,1]（以适配老的 save_ab_grid 逻辑）
    """
    device = batch["seg"].device
    seg = batch["seg"].clamp(0,1).to(device)
    prompts = batch["prompt"]

    # 构造 cond（与训练一致）
    c_cross = textenc.encode(prompts)  # [B,77,768]
    cond = {"c_concat": [seg], "c_crossattn": [c_cross]}
    uc = None
    cfg_scale = scale

    # 采样器
    sampler = getattr(model, "sampler", None)
    if sampler is None:
        from ControlNet.ldm.models.diffusion.ddim import DDIMSampler
        sampler = DDIMSampler(model)

    B, _, H, W = seg.shape
    z_shape = (4, H//8, W//8)  # 与训练一致
    samples, _ = sampler.sample(
        S=steps,
        conditioning=cond,
        batch_size=B,
        shape=z_shape,
        verbose=False,
        unconditional_guidance_scale=cfg_scale,
        unconditional_conditioning=uc,
        eta=eta
    )

    # 解码到 [0,1]，再拉回 [-1,1] 以兼容老可视化
    if hasattr(model, "decode_first_stage"):
        x01 = model.decode_first_stage(samples)
        x01 = (x01.clamp(-1,1) + 1.0) / 2.0  # 万一 decode 返回的是[-1,1]，先规整
        x01 = x01.clamp(0,1)
    else:
        x01 = samples.clamp(0,1)

    x_m11 = x01 * 2.0 - 1.0
    return x_m11


if __name__ == "__main__":
    main()

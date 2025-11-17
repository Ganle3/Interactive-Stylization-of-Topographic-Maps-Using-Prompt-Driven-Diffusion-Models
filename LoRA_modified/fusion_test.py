# -*- coding: utf-8 -*-
# === Self-contained inference preview for CN-LoRA (no training imports, no side effects) ===
import os, sys, math, json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
import numpy as np

# --------------------------
# USER PATHS (edit these 5)
# --------------------------
CKPT = r"D:\Junyhuang\Project2\BaseModel\Swisstopo.ckpt"
CFG  = r"D:\Junyhuang\Project2\ControlNet\models\cldm_v15.yaml"
LORA_PT = r"D:\Junyhuang\Project2\Outputs_overfit\Sameseg_multiprompt\batch_switch_lora_Com\unet_lora_step002500.pt"
TEXT_ADAPTER_PT = r"D:\Junyhuang\Project2\Outputs_overfit\Sameseg_multiprompt\batch_switch_lora_Com\textenc_adapter_step002500.pt"
JSONL = r"D:\Junyhuang\Project2_Data\Training Data\Sameseg_diffitem_8\meta\pairs_seggroup.jsonl"
ROOTDIR = r"D:\Junyhuang\Project2_Data\Training Data\Sameseg_diffitem_8"
OUT_PNG = r"D:\Junyhuang\Project2\Outputs_overfit\Sameseg_multiprompt\fusion_test_result.png"

IMAGE_SIZE = 512  # keep 512 as you trained
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- only add ControlNet root to import the model class; we will NOT import any training utilities
sys.path.insert(0, r"D:\Junyhuang\Project2\ControlNet")
from ControlNet.ldm.util import instantiate_from_config

# --------------------------
# Minimal, side-effect-free dataset
# --------------------------
class PairsJSONLDatasetLite:
    def __init__(self, jsonl_path, rootdir, image_size=512, seg_mode="RGB"):
        self.root = Path(rootdir)
        self.items = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                j = json.loads(line)
                self.items.append({
                    "prompt": j["prompt"],
                    "seg": (self.root / j["source"]).as_posix(),
                    "gt":  (self.root / j["target"]).as_posix()
                })
        self.size = image_size
        self.seg_mode = seg_mode

    def __len__(self):
        return len(self.items)

    def _load_img(self, path, mode="RGB"):
        im = Image.open(path).convert(mode)
        # resize to IMAGE_SIZE using nearest for seg, bilinear for gt
        if mode == "RGB":  # seg or gt both RGB in your pipeline
            interp = Image.NEAREST if 'seg' in os.path.basename(path).lower() else Image.BILINEAR
        else:
            interp = Image.BILINEAR
        im = im.resize((self.size, self.size), interp)
        arr = np.array(im).astype(np.float32) / 127.5 - 1.0  # [-1,1]
        # HWC->CHW
        arr = np.transpose(arr, (0,1,2))
        if arr.ndim == 2:
            arr = np.expand_dims(arr, -1)
        arr = np.transpose(arr, (2,0,1))
        return torch.from_numpy(arr)

    def __getitem__(self, idx):
        it = self.items[idx]
        seg = self._load_img(it["seg"], "RGB")
        gt  = self._load_img(it["gt"],  "RGB")
        return {"prompt": it["prompt"], "seg": seg, "gt": gt}

# --------------------------
# CN-LoRA: LoRA wrapper + condition injection (no training imports)
# --------------------------
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
def set_lora_condition(unet: nn.Module, seg_feat: torch.Tensor, text_feat: torch.Tensor):
    for m in unet.modules():
        if isinstance(m, LoRALinear):
            m.set_control_feature(seg_feat, text_feat)

# --------------------------
# Minimal DDIM preview (no training imports)
# --------------------------
from TextEncoder_Finetuning.vis_metrics import sample_preview_CNLora
# --------------------------
# Save grid (seg | pred | gt | prompt)
# --------------------------
def _to_img(x):  # [-1,1] CHW -> uint8 HWC
    x = (x.clamp(-1,1) + 1.0) * 127.5
    x = x.permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
    return Image.fromarray(x)

import torchvision.utils as vutils
def to_vis(x):
    # 训练/解码域 [-1,1] -> 显示域 [0,1]
    return ((x + 1) / 2).clamp(0, 1) if x.min() < 0 else x.clamp(0, 1)

def save_concept_grid_new(seg_list, preds_list, gts_list, prompts, out_png, padding=2):
    """
    seg_list:  List[[1,C,H,W]] 长度 K
    preds_list: List[[1,C,H,W]] 长度 K
    gts_list:   List[[1,C,H,W]] 长度 K
    """
    K = len(preds_list)
    assert K == len(gts_list) == len(seg_list) and K > 0

    segs  = torch.cat(seg_list,  dim=0)  # [K,C,H,W]
    preds = torch.cat(preds_list, dim=0)
    gts   = torch.cat(gts_list,   dim=0)

    grid_seg  = vutils.make_grid(to_vis(segs),  nrow=K, padding=padding)
    grid_pred = vutils.make_grid(to_vis(preds), nrow=K, padding=padding)
    grid_gt   = vutils.make_grid(to_vis(gts),   nrow=K, padding=padding)

    # 垂直拼接三行：seg / pred / gt
    full = torch.cat([grid_seg, grid_pred, grid_gt], dim=1)
    vutils.save_image(full, out_png)
    print(f"[viz] saved {out_png} (K={K})")

# --------------------------
# MAIN
# --------------------------
def main():
    # 1) load base model
    cfg = OmegaConf.load(CFG)
    model = instantiate_from_config(cfg.model)
    sd = torch.load(CKPT, map_location="cpu")
    sd = sd.get("state_dict", sd)
    model.load_state_dict(sd, strict=False)
    model.to(DEVICE).eval()

    # 2) LoRA injection and load weights
    lora_wrapped = lora_qkv_combine(model.model.diffusion_model, r=12)
    lora_sd = torch.load(LORA_PT, map_location=DEVICE)
    model.model.diffusion_model.load_state_dict(lora_sd, strict=False)

    # 3) text encoder: reuse model.cond_stage_model + load adapter (no extra BERT init)
    textenc = model.cond_stage_model
    te_sd = torch.load(TEXT_ADAPTER_PT, map_location=DEVICE)
    textenc.load_state_dict(te_sd, strict=False)
    textenc.eval()

    # 4) dataset lite (NO training imports)
    ds = PairsJSONLDatasetLite(JSONL, ROOTDIR, image_size=IMAGE_SIZE, seg_mode="RGB")
    idxs = [0, 9, 18, 27, 36, 45, 54, 63]
    segs, preds, gts, proms = [], [], [], []

    with torch.no_grad():
        for i in idxs:
            vb = ds[i]
            seg = vb["seg"].unsqueeze(0).to(DEVICE, torch.float32)
            gt  = vb["gt"].unsqueeze(0).to(DEVICE, torch.float32)
            prompt = [vb["prompt"]]

            # prepare control features and set to all LoRALinear
            seg_feat  = F.adaptive_avg_pool2d(seg, 1).flatten(1)     # [1,C]
            text_feat = textenc.encode(prompt)                        # [1,L,768] or [1,768]
            set_lora_condition(model.model.diffusion_model, seg_feat, text_feat)

            pred = sample_preview_CNLora(model, textenc,
                                         {"seg": seg, "gt": gt, "prompt": prompt},
                                         DEVICE, steps=12, scale=7.5, eta=0.0, seed=1234)

            segs.append(seg); preds.append(pred); gts.append(gt); proms.append(prompt[0])

    save_concept_grid_new(segs, preds, gts, proms, OUT_PNG)
    print(f"[DONE] Saved visualization → {OUT_PNG}")

if __name__ == "__main__":
    # hard-kill any cached training modules if the interpreter had them
    for k in list(sys.modules.keys()):
        if any(s in k for s in ["ControlNet_LoRA", "LoRA_utils", "TextEncoder_Finetuning.train", "trainer", "train_"]):
            sys.modules.pop(k, None)
    main()
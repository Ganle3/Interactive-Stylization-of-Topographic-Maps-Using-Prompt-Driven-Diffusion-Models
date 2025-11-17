# vis_metrics.py
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from torch.amp import autocast
import os
from TextEncoder_Finetuning.token_utils import BertAlign, build_cond_uc_safe_pad

import torchvision.utils as vutils

def to_vis(x):
    # 训练/解码域 [-1,1] -> 显示域 [0,1]
    return ((x + 1) / 2).clamp(0, 1) if x.min() < 0 else x.clamp(0, 1)

def save_concept_grid(seg_1, preds_list, gts_list, prompts, out_png, padding=2):
    """
    seg_1: [1,C,H,W] 固定一张 seg
    preds_list: List[[1,C,H,W]] 长度 K
    gts_list:   List[[1,C,H,W]] 长度 K
    """
    K = len(preds_list)
    assert K == len(gts_list) and K > 0

    preds = torch.cat(preds_list, dim=0)         # [K,C,H,W]
    gts   = torch.cat(gts_list,   dim=0)         # [K,C,H,W]

    grid_pred = vutils.make_grid(to_vis(preds), nrow=K, padding=padding)  # [C,H, K*W+…]
    grid_gt   = vutils.make_grid(to_vis(gts),   nrow=K, padding=padding)

    # 关键：把 seg 复制 K 次，做成同宽的“seg 行”
    seg_row   = seg_1.expand(K, -1, -1, -1)      # [K,C,H,W]
    grid_seg  = vutils.make_grid(seg_row, nrow=K, padding=padding)

    # 现在三行宽度一致，可以按高度拼接
    full = torch.cat([grid_seg, grid_pred, grid_gt], dim=1)  # dim=1 -> 高度方向
    vutils.save_image(full, out_png)
    print(f"[viz] saved {out_png} (K={K})")


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


class MeterEMA:
    """simple EMA smoother metics, use to display training loss"""
    def __init__(self, beta=0.98):
        self.beta = beta
        self.avg = None

    def update(self, x: float) -> float:
        self.avg = x if self.avg is None else self.beta * self.avg + (1 - self.beta) * x
        return self.avg

@torch.no_grad()
def validate_step(model, textenc, batch, device, max_length=77, iters=64, fixed_t=None, fixed_noise=None):
    """
    Do a light epsilon-MSE evaluation in validation set:
      - randomly sample t, calculate average noise regress MSE
      - no completely sampling
    return：val_mse(float)
    """
    seg = batch["seg"].to(device=device, dtype=torch.float32)
    gt  = batch["gt"].to(device=device, dtype=torch.float32)
    prompts = batch["prompt"]

    # encode to latent
    z = model.get_first_stage_encoding(model.encode_first_stage(gt))
    z = z.to(dtype=torch.float32)
    B = z.size(0)

    c_cross = textenc.encode(prompts).to(device=device, dtype=torch.float32)
    cond = {"c_crossattn": [c_cross], "c_concat": [seg]}

    loss_total = 0.0

    if fixed_t is None or fixed_noise is None:
        g = torch.Generator(device=device).manual_seed(12345)

    for _ in range(iters):
        if fixed_t is not None:
            t = fixed_t
        else:
            t = torch.randint(0, model.num_timesteps, (B,), generator=g, device=device).long()

        if fixed_noise is not None:
            noise = fixed_noise
        else:
            noise = torch.randn_like(z)

        z_noisy = model.q_sample(z, t, noise=noise)

        with autocast('cuda', enabled=(device == 'cuda'), dtype=torch.float32):
            eps_hat = model.apply_model(z_noisy, t, cond)

        loss_total += F.mse_loss(eps_hat.float(), noise.float()).item()

    return loss_total / float(iters)


@torch.no_grad()
def _ssim_batch(x, y, C1=0.01**2, C2=0.03**2):
    """
    x,y: [B,3,H,W] in [0,1]
    简化版 SSIM（3x3 平均池化近似），够用来监控趋势
    """
    import torch.nn.functional as F
    pad = 1
    mu_x = F.avg_pool2d(x, 3, 1, pad)
    mu_y = F.avg_pool2d(y, 3, 1, pad)
    sigma_x  = F.avg_pool2d(x*x, 3, 1, pad) - mu_x*mu_x
    sigma_y  = F.avg_pool2d(y*y, 3, 1, pad) - mu_y*mu_y
    sigma_xy = F.avg_pool2d(x*y, 3, 1, pad) - mu_x*mu_y
    num = (2*mu_x*mu_y + C1) * (2*sigma_xy + C2)
    den = (mu_x*mu_x + mu_y*mu_y + C1) * (sigma_x + sigma_y + C2)
    ssim_map = num / (den + 1e-12)
    return ssim_map.mean(dim=[1,2,3])  # [B]


@torch.no_grad()
def validate_image_metrics(model, textenc, batch, device, steps=15, scale=7.5, eta=0.0):
    """
    解码一遍 → 计算图像空间指标（L1/SSIM）。返回 dict: {"l1": float, "ssim": float}
    """
    # 采样一遍
    preds = sample_preview(
        model, textenc, batch, device,
        steps=steps, scale=scale, eta=eta, seed=1234
    )  # [-1,1]
    preds = (preds + 1) / 2.0
    gts   = (batch["gt"].to(device=device, dtype=torch.float32) + 1) / 2.0
    preds = torch.clamp(preds, 0.0, 1.0)
    gts   = torch.clamp(gts,   0.0, 1.0)

    # L1
    l1 = torch.abs(preds - gts).mean(dim=[1,2,3])  # [B]
    # SSIM
    ssim = _ssim_batch(preds, gts)

    return {
        "l1": float(l1.mean().item()),
        "ssim": float(ssim.mean().item())
    }


@torch.no_grad()
def save_triplet_grid(seg_batch, pred_batch, gt_batch, out_png, nrow=3, prompts=None):
    """
    save (seg, pred, gt) grid picture:
      - 每个样本一行、3 列固定：[SEG | PRED | GT]
      - 可选在每行上方写出对应 prompt 文本
      - seg: 0..1（C,H,W）
      - pred/gt: -1..1（C,H,W）-- 内部映射到 0..1
    """
    import torchvision.transforms.functional as TF
    from PIL import ImageDraw, ImageFont

    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    segs = seg_batch.detach().float().cpu()           # (B,C,H,W), 0..1
    gts  = (gt_batch.detach().float().cpu() + 1) / 2  # → 0..1
    preds = (pred_batch.detach().float().cpu() + 1) / 2 if pred_batch is not None else None

    B, C, H, W = segs.shape
    tiles = []
    for i in range(B):
        if preds is not None:
            tiles += [segs[i], preds[i], gts[i]]
        else:
            tiles += [segs[i], gts[i]]

    #  3 col (SEG|PRED|GT), one sample one row
    grid = make_grid(tiles, nrow=3, padding=2)
    save_image(grid, out_png)  # 先保存纯图

    # 叠加每行的 prompt 文本
    if prompts is not None:
        img = TF.to_pil_image(grid)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        pad = 2
        cell_h = H
        row_top = pad
        for i in range(B):
            text = str(prompts[i])
            rect_h = 16
            # 半透明黑底条提升可读性
            draw.rectangle([(0, max(0, row_top - rect_h - 1)), (img.width, row_top - 1)], fill=(0, 0, 0, 127))
            draw.text((4, row_top - rect_h), text, fill=(255, 255, 255), font=font)
            row_top += cell_h + pad  # 下一行

        img.save(out_png)  # replace save (with text)
# def save_triplet_grid(seg_batch, pred_batch, gt_batch, out_png, nrow=4):
#     """
#     Stitch several (seg, pred, gt) to a huge graph to save
#     - seg: 0..1（C,H,W）
#     - pred/gt: -1..1（C,H,W）-- inside mapping back to 0..1
#     """
#     os.makedirs(os.path.dirname(out_png), exist_ok=True)
#
#     segs = seg_batch.detach().float().cpu()
#     gts  = (gt_batch.detach().float().cpu() + 1) / 2.0
#
#     if pred_batch is not None:
#         preds = (pred_batch.detach().float().cpu() + 1) / 2.0
#         tiles = []
#         for s, p, g in zip(segs, preds, gts):
#             tiles += [s, p, g]
#         grid = make_grid(tiles, nrow=nrow, padding=2)
#     else:
#         tiles = []
#         for s, g in zip(segs, gts):
#             tiles += [s, g]
#         grid = make_grid(tiles, nrow=nrow, padding=2)
#
#     save_image(grid, out_png)


# @torch.no_grad()
# def save_ab_grid(seg_batch, pred_a, pred_b, gt_batch, out_png, prompts_a):
#     """
#     每个样本一行四列：SEG | PRED(A) | PRED(B) | GT，并在行上方写 A 的 prompt
#     B 固定标注为 (empty prompt)
#     """
#     import torchvision.transforms.functional as TF
#     from PIL import ImageDraw, ImageFont
#
#     os.makedirs(os.path.dirname(out_png), exist_ok=True)
#     segs = seg_batch.detach().float().cpu()
#     gts  = (gt_batch.detach().float().cpu() + 1) / 2
#     pa   = (pred_a.detach().float().cpu() + 1) / 2
#     pb   = (pred_b.detach().float().cpu() + 1) / 2
#
#     B, C, H, W = segs.shape
#     tiles = []
#     for i in range(B):
#         tiles += [segs[i], pa[i], pb[i], gts[i]]
#
#     grid = make_grid(tiles, nrow=4, padding=2)
#     save_image(grid, out_png)
#
#     # 叠字：A 的 prompt + 固定 B 说明
#     img = TF.to_pil_image(grid)
#     draw = ImageDraw.Draw(img)
#     try:
#         font = ImageFont.load_default()
#     except Exception:
#         font = None
#
#     pad = 2
#     row_top = pad
#     for i in range(B):
#         text = f"A: {prompts_a[i]}   |   B: (empty prompt)"
#         rect_h = 16
#         draw.rectangle([(0, max(0, row_top - rect_h - 1)), (img.width, row_top - 1)], fill=(0,0,0,127))
#         draw.text((4, row_top - rect_h), text, fill=(255,255,255), font=font)
#         row_top += H + pad
#
#     img.save(out_png)

@torch.no_grad()
def save_ab_grid(seg_batch, pred_a, pred_b, gt_batch, out_png, prompts_a, pred_b0=None):
    """
    每个样本一行：
      若 pred_b0 is None: 4 列 => [SEG | PRED(A) | PRED(B) | GT]
      若 pred_b0 存在:     5 列 => [SEG | PRED(A) | PRED(B) | PRED(B0) | GT]
    行首叠字：A 的 prompt + 'B: no text, CFG=1' +（若有）'B0: no text, LoRA off'
    """
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    segs = seg_batch.detach().float().cpu()
    gts  = (gt_batch.detach().float().cpu() + 1) / 2
    pa   = (pred_a.detach().float().cpu() + 1) / 2
    pb   = (pred_b.detach().float().cpu() + 1) / 2
    if pred_b0 is not None:
        pb0 = (pred_b0.detach().float().cpu() + 1) / 2

    B, C, H, W = segs.shape
    tiles = []
    ncol = 5 if pred_b0 is not None else 4
    for i in range(B):
        row = [segs[i], pa[i], pb[i]]
        if pred_b0 is not None:
            row.append(pb0[i])
        row.append(gts[i])
        tiles += row

    grid = make_grid(tiles, nrow=ncol, padding=2)
    save_image(grid, out_png)

    # 叠文字说明
    import torchvision.transforms.functional as TF
    from PIL import ImageDraw, ImageFont

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

from LoRA_modified.LoRA_utils import LoRALinear
def set_lora_condition(root: torch.nn.Module, seg_feat: torch.Tensor, text_tokens: torch.Tensor):
    """
    仅给 LoRALinear 实例注入条件；不会碰到 MemoryEfficientCrossAttention。
    """
    for m in root.modules():
        if isinstance(m, LoRALinear):
            m.set_control_feature(seg_feat, text_tokens)

@torch.no_grad()
def sample_preview_CNLora(model, textenc, batch, device, steps=30, scale=7.5, eta=0.0, seed=1234):
    """
    用 DDIM 采样若干步生成预测图，返回 pred_imgs in [-1,1]
    - seg（ControlNet hint）用 fp32
    - 文本条件用 fp16
    - 自动注入 LoRA 条件（seg_feat + text_feat）
    """
    from ControlNet.ldm.models.diffusion.ddim import DDIMSampler

    model.eval()
    textenc.eval()

    seg = batch["seg"].to(device=device, dtype=torch.float32)
    gt  = batch["gt"].to(device=device, dtype=torch.float32)
    prompts = batch["prompt"]
    B, _, H, W = seg.shape

    c_cross  = textenc.encode(prompts).to(device=device, dtype=torch.float16)
    uc_cross = torch.zeros_like(c_cross)

    with torch.no_grad():
        emb = torch.zeros(B, model.control_model.model_channels * 4, device=device, dtype=seg.dtype)
        control_states = model.control_model.input_hint_block(seg, emb, context=None)
        if isinstance(control_states, (tuple, list)):
            control_states = control_states[-1]
        seg_feat = F.adaptive_avg_pool2d(control_states, 1).flatten(1)

    print(f"[sample_preview] seg_feat={tuple(seg_feat.shape)}, text_feat={tuple(c_cross.shape)}")

    set_lora_condition(model.model.diffusion_model, seg_feat, c_cross)

    cond = {"c_crossattn": [c_cross],  "c_concat": [seg]}
    uc   = {"c_crossattn": [uc_cross], "c_concat": [seg]}

    sampler = DDIMSampler(model)
    torch.manual_seed(seed)
    z_shape = (4, H // 8, W // 8)
    print(f"[DDIM] start sampling with z_shape={z_shape}, steps={steps}")

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

    pred_imgs = model.decode_first_stage(samples.to(torch.float32))
    pred_imgs = torch.clamp(pred_imgs, -1.0, 1.0)

    print("[sample_preview] sampling done.")
    return pred_imgs



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

    # 文本条件（fp16）
    c_cross  = textenc.encode(prompts).to(device=device, dtype=torch.float32)
    uc_cross = torch.zeros_like(c_cross)

    cond = {"c_crossattn": [c_cross],  "c_concat": [seg]}
    uc   = {"c_crossattn": [uc_cross], "c_concat": [seg]}

    sampler = DDIMSampler(model)
    torch.manual_seed(seed)

    z_shape = (4, H // 8, W // 8)

    # 把整个采样过程放进 autocast(fp16)
    with autocast('cuda', enabled=(device == 'cuda'), dtype=torch.float32):
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
# @torch.no_grad()
# def sample_preview(model, textenc, batch, device,
#                    steps=30, scale=7.5, eta=0.0, seed=1234,
#                    use_text: bool = True,        # A/B 开关
#                    disable_lora: bool = False):   # B0 用到
#     from ControlNet.ldm.models.diffusion.ddim import DDIMSampler
#
#     model.eval()
#     seg = batch["seg"].to(device=device, dtype=torch.float32)
#     gt  = batch["gt"].to(device=device, dtype=torch.float32)
#     prompts = batch["prompt"]
#     B, _, H, W = seg.shape
#
#     # --- 文本条件：与训练一致，只用 BERT；不叠加 CLIP ---
#     if use_text:
#         c_cross = textenc.encode(prompts).to(device=device)  # B×77×768 (BERT)
#         uc_cross = textenc.encode([""] * len(prompts)).to(device=device)
#         cfg_scale = scale
#     else:
#         # 纯无文本：同一编码器的空提示；CFG 置 1.0
#         c_cross = textenc.encode([""] * len(prompts)).to(device=device)
#         uc_cross = c_cross
#         cfg_scale = 1.0
#
#     cond = {"c_crossattn": [c_cross], "c_concat": [seg]}
#     uc = {"c_crossattn": [uc_cross], "c_concat": [seg]}
#     print("c_cross:", tuple(c_cross.shape), "uc_cross:", tuple(uc_cross.shape))
#
#     print("[DBG] use_text=", use_text)
#     print("[DBG] seg dtype/min/max:", seg.dtype, seg.min().item(), seg.max().item())
#     # # 如果 use_text=False，显示创建的 c_cross 形状
#     # if not use_text:
#     #     print("[DBG] c_cross shape:", c_cross.shape, "norm:", c_cross.float().abs().mean().item())
#     # else:
#     #     print("[DBG] real c_cross shape:", c_cross.shape, "norm:", c_cross.float().abs().mean().item())
#     # # 检查 ControlNet 是否被冻结
#     # frozen_bad = [n for n, p in model.named_parameters() if
#     #               (n.startswith("control_model.") or n.startswith("control_")) and p.requires_grad]
#     # print("[DBG] unfrozen control params count:", len(frozen_bad))
#
#     sampler = DDIMSampler(model)
#     torch.manual_seed(seed)
#     z_shape = (4, H // 8, W // 8)
#
#     # 采样时临时关闭 LoRA（做 B0）
#     _lora_tmp = []
#     if disable_lora:
#         for m in model.model.diffusion_model.modules():
#             for attr in ("to_q", "to_k", "to_v"):
#                 lin = getattr(m, attr, None)
#                 if hasattr(lin, "scaling"):
#                     _lora_tmp.append((lin, lin.scaling)); lin.scaling = 0.0
#             if hasattr(m, "to_out") and hasattr(m.to_out, "__getitem__"):
#                 lin = m.to_out[0]
#                 if hasattr(lin, "scaling"):
#                     _lora_tmp.append((lin, lin.scaling)); lin.scaling = 0.0
#
#     with autocast('cuda', enabled=(device == 'cuda'), dtype=torch.float16):
#         samples, _ = sampler.sample(
#             S=steps, conditioning=cond, batch_size=B, shape=z_shape, verbose=False,
#             unconditional_guidance_scale=cfg_scale, unconditional_conditioning=uc, eta=eta, x_T=None
#         )
#
#     if _lora_tmp:
#         for lin, s in _lora_tmp:
#             lin.scaling = s
#
#     pred_imgs = model.decode_first_stage(samples.to(torch.float32))
#     pred_imgs = torch.clamp(pred_imgs, -1.0, 1.0)
#     return pred_imgs


#  utils for token joint preview
def make_cond_uc_for_mode(
    seg, prompts, model, clip_textenc, bert_textenc, device,
    clip_style_text="map in swisstopo style", mode="main",
    gamma=1.0, w_clip=0.4, bert_align=None, start_idx=8
):
    """
    mode:
      - "main":    cond = w_clip*clip + gamma*bert；uc = w_clip*clip
      - "bert":    cond = gamma*bert；            uc = 0（BERT-only）
      - "clipwk":  cond = 0.30*clip + gamma*bert；uc = 0.30*clip
    """
    if mode == "main":
        return build_cond_uc_safe_pad(
            seg, prompts, model, clip_textenc, bert_textenc, device,
            clip_style_text=clip_style_text, gamma=gamma, w_clip=w_clip,
            bert_align=bert_align, start_idx=start_idx
        )

    # 拿一个 clip_ctx
    mp_dtype = next(model.parameters()).dtype
    seg = seg.to(device=device, dtype=mp_dtype)
    B = seg.size(0)
    with torch.no_grad():
        clip_ctx = clip_textenc.get_learned_conditioning([clip_style_text] * B).to(device=device, dtype=mp_dtype)

    # 复用 BERT tail
    bert_tok = bert_textenc.encode(prompts).to(device=device, dtype=mp_dtype)
    if bert_align is not None:
        bert_tok = bert_align(bert_tok)
    bert_tok = F.layer_norm(bert_tok, (bert_tok.size(-1),))
    max_len = clip_ctx.size(1)
    tail = torch.zeros(B, max_len, clip_ctx.size(-1), device=device, dtype=mp_dtype)
    end = min(max_len, start_idx + bert_tok.size(1))
    if end > start_idx:
        tail[:, start_idx:end, :] = bert_tok[:, :end - start_idx, :]

    if mode == "bert":
        cond_ctx = gamma * tail
        uc_ctx   = torch.zeros_like(cond_ctx)
    elif mode == "clipwk":
        weak = 0.30
        cond_ctx = weak * clip_ctx + gamma * tail
        uc_ctx   = weak * clip_ctx
    else:
        raise ValueError(f"Unknown mode: {mode}")

    cond = {"c_concat": [seg], "c_crossattn": [cond_ctx]}
    uc   = {"c_concat": [seg], "c_crossattn": [uc_ctx]}
    return cond, uc


@torch.no_grad()
def preview_three_modes(
    model, textenc, seg, prompts, device,
    steps=25, scale=4.0, eta=0.0, seed=1234,
    clip_style_text="map in swisstopo style",
    gamma=1.0, w_clip=0.4, bert_align=None, start_idx=8
):
    """返回 (preds_main, preds_bertonly, preds_clipweak)，都在 [-1,1]"""
    from ControlNet.ldm.models.diffusion.ddim import DDIMSampler

    mp_dtype = next(model.parameters()).dtype
    seg = seg.to(dtype=mp_dtype)

    B, _, H, W = seg.shape
    z_shape = (4, H // 8, W // 8)
    sampler = DDIMSampler(model)

    def sample_with(mode):
        cond, uc = make_cond_uc_for_mode(
            seg, prompts, model, model, textenc, device,
            clip_style_text=clip_style_text, mode=mode,
            gamma=gamma, w_clip=w_clip, bert_align=bert_align, start_idx=start_idx
        )
        torch.manual_seed(seed)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(device == 'cuda')):
            samples, _ = sampler.sample(
                S=steps, conditioning=cond, batch_size=B, shape=z_shape, verbose=False,
                unconditional_guidance_scale=scale, unconditional_conditioning=uc, eta=eta, x_T=None
            )
        imgs = model.decode_first_stage(samples.to(dtype=torch.float32))
        return torch.clamp(imgs, -1.0, 1.0)

    preds_main     = sample_with("main")
    preds_bertonly = sample_with("bert")
    preds_clipweak = sample_with("clipwk")
    return preds_main, preds_bertonly, preds_clipweak


@torch.no_grad()
def save_abc_grid(seg, imgs_a, imgs_b, imgs_c, gt, out_png):
    """
    显示 5 列：seg | A(main) | B(bert-only) | C(clip-weak) | gt
    约定：
      - seg: 已是 [0,1]
      - imgs_a/b/c, gt: 在 [-1,1]，这里转 [0,1]
    """
    import os
    import torchvision.utils as vutils

    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    def to01_from_m1p1(x):
        return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)

    B = seg.size(0)
    rows = []
    for i in range(B):
        # seg 只做 dtype/device 对齐，不做数值变换
        seg01 = seg[i].to(dtype=imgs_a.dtype, device=imgs_a.device)
        seg01 = torch.clamp(seg01, 0.0, 1.0)

        A01 = to01_from_m1p1(imgs_a[i])
        B01 = to01_from_m1p1(imgs_b[i])
        C01 = to01_from_m1p1(imgs_c[i])
        gt01 = to01_from_m1p1(gt[i].to(dtype=imgs_a.dtype, device=imgs_a.device))

        row = torch.stack([seg01, A01, B01, C01, gt01], dim=0)  # [5,3,H,W]
        rows.append(row)

    grid = torch.cat(rows, dim=0)  # [B*5,3,H,W]
    grid_img = vutils.make_grid(grid, nrow=5, padding=2)
    vutils.save_image(grid_img, out_png)



@torch.no_grad()
def sample_preview_new(
    model, textenc, batch, device,
    steps=30, scale=7.5, eta=0.0, seed=1234, clip_style_text="map in swisstopo style"
):
    """
    用 DDIM 采样若干步生成预测图，返回 pred_imgs in [-1,1]
    - seg（ControlNet hint）用 fp32（或与模型主 dtype 一致）
    - 文本条件/采样过程放在 autocast(fp16) 下
    """
    from ControlNet.ldm.models.diffusion.ddim import DDIMSampler

    model.eval()
    seg = batch["seg"].to(device=device, dtype=torch.float32)
    gt  = batch["gt"].to(device=device, dtype=torch.float32)  # 仅用来拿尺寸
    prompts = batch["prompt"]
    B, _, H, W = seg.shape


    # latent 尺寸（常见 SD1.5 为 8x 下采样）
    z_shape = (4, H // 8, W // 8)

    # 统一混精度 dtype
    mp_dtype = next(model.parameters()).dtype

    # 构造 cond / uc（关键：c_crossattn 为单元素列表）
    cond, uc = build_cond_uc_safe_pad(
        seg=seg, prompts=prompts,
        model=model,                # 用 model.get_learned_conditioning 取 CLIP 基底
        clip_textenc=model,
        bert_textenc=textenc,       # 你的 BERT 编码器
        device=device,
        clip_style_text=clip_style_text,
    )

    sampler = DDIMSampler(model)
    torch.manual_seed(seed)

    # 采样置于 autocast(fp16) 下
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(device == 'cuda')):
        samples, _ = sampler.sample(
            S=steps,
            conditioning=cond,
            batch_size=B,
            shape=z_shape,
            verbose=False,
            unconditional_guidance_scale=scale,   # 建议 4.5–6.0 更稳
            unconditional_conditioning=uc,
            eta=eta,
            x_T=None,
        )

    # 解码可用 fp32 更稳
    pred_imgs = model.decode_first_stage(samples.to(dtype=torch.float32))
    pred_imgs = torch.clamp(pred_imgs, -1.0, 1.0)
    return pred_imgs

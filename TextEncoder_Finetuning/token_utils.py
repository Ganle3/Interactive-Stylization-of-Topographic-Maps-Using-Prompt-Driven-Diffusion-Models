import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_clip_tokenizer(clip_textenc):
    # 兼容你工程里 model/cond_stage_model 的两种写法
    if hasattr(clip_textenc, "cond_stage_model") and hasattr(clip_textenc.cond_stage_model, "tokenizer"):
        return clip_textenc.cond_stage_model.tokenizer
    if hasattr(clip_textenc, "tokenizer"):
        return clip_textenc.tokenizer
    raise RuntimeError("Cannot find CLIP tokenizer on clip_textenc")


@torch.no_grad()
def _clip_ctx_and_after_eos(clip_textenc, clip_prompts, device, max_len=77, dtype=None):
    """返回 CLIP 上下文 [B,77,768] 以及每个样本可写入 BERT 的起始索引 start = first_eos+1。"""
    if dtype is None:
        dtype = next(clip_textenc.parameters()).dtype if hasattr(clip_textenc, "parameters") else torch.float16
    clip_ctx = clip_textenc.get_learned_conditioning(clip_prompts).to(device=device, dtype=dtype)  # [B,77,768]

    tok = _get_clip_tokenizer(clip_textenc)
    enc = tok(clip_prompts, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    ids = enc["input_ids"].to(device)
    eos_id = getattr(tok, "eos_token_id", 49407)
    # 每行第一个 eos 位置（注意：SD 的 tokenizer 会用 eos 填充到 77）
    first_eos = (ids == eos_id).float().argmax(dim=1)  # [B]
    start = torch.clamp(first_eos + 1, max=max_len - 1)  # 至少保留一个 EOS，本次从它后面开始写
    return clip_ctx, start  # [B,77,768], [B]


def fuse_clip_bert_in_pad(
    clip_textenc, bert_textenc,
    clip_prompts, bert_prompts,
    device, gamma=0.25, uc_mix=0.15,
    bert_align=None, max_len=77, dtype=None
):
    """
    - clip_prompts: 用于 CLIP 基底的文本（比如固定 'map in swisstopo style'）
    - bert_prompts: 你想让 BERT 学的增量文本（比如 'set forest to dark green color'）
    - gamma: BERT 残差强度（0.2~0.3 起步）
    - uc_mix: 软 CFG：无条件分支也混入一点点 BERT（0.1~0.2）
    返回 cond_ctx/uc_ctx，形状都为 [B,77,768]
    """
    if dtype is None:
        dtype = next(clip_textenc.parameters()).dtype if hasattr(clip_textenc, "parameters") else torch.float16

    # 1) CLIP 基底 + 每样本可写入的起始位置
    clip_ctx, start = _clip_ctx_and_after_eos(clip_textenc, clip_prompts, device, max_len=max_len, dtype=dtype)  # [B,77,768], [B]
    B = clip_ctx.size(0)

    # 2) BERT 侧（你已有 adapter/align 就在 encode 里）
    bert_ctx = bert_textenc.encode(bert_prompts).to(device=device, dtype=dtype)      # [B,Lb,768]
    if bert_align is not None:
        bert_ctx = bert_align(bert_ctx)                                              # 可选的小残差/对齐
    bert_ctx = F.layer_norm(bert_ctx, (bert_ctx.size(-1),))                          # 稳定一点
    bert_ctx = gamma * bert_ctx                                                      # 控制强度

    # 3) cond：把 BERT 写进 [start: ) 的 padding 槽位；长度保持 77
    cond_ctx = clip_ctx.clone()
    for i in range(B):
        s = int(start[i].item())
        k = min(max_len - s, bert_ctx.size(1))
        if k > 0:
            cond_ctx[i, s:s+k, :] = bert_ctx[i, :k, :]

    # 4) uc：软 CFG，无条件也混入一点 BERT
    with torch.no_grad():
        uc_clip = clip_textenc.get_learned_conditioning([""] * B).to(device=device, dtype=dtype)
    uc_ctx = uc_clip.clone()
    if uc_mix > 0.0:
        for i in range(B):
            s = int(start[i].item())
            k = min(max_len - s, bert_ctx.size(1))
            if k > 0:
                uc_ctx[i, s:s+k, :] = uc_ctx[i, s:s+k, :] + uc_mix * bert_ctx[i, :k, :]

    return cond_ctx, uc_ctx  # [B,77,768] x2



# --- 长度对齐：把 [B, Lb, 768] → [B, 77, 768] ---
class TokenAlign77(nn.Module):
    def __init__(self, target_len=77):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(target_len)  # 无参数，稳定

    def forward(self, x):          # x: [B, Lb, 768]
        x = x.transpose(1, 2)      # [B, 768, Lb]
        x = self.pool(x)           # [B, 768, 77]
        x = x.transpose(1, 2)      # [B, 77, 768]
        return x


class BertAlign(nn.Module):
    """
    只对 BERT 的 token 表征做轻量对齐；绝不改动 CLIP 的 77 个 token。
    默认零初始化 + 门控，避免一上来破坏分布。
    """
    def __init__(self, dim=768, gate_init=-2.0):  # sigmoid(-2) ≈ 0.12
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.beta = nn.Parameter(torch.tensor(gate_init))  # 可学习标量门控

    def forward(self, x):
        # 只 LN BERT 路径以稳定训练；不要 LN CLIP 路径
        x = F.layer_norm(x, (x.size(-1),))
        x = self.proj(x)
        x = torch.sigmoid(self.beta) * x  # 小残差
        return x


def build_cond_uc_safe_pad(
    seg, prompts, model, clip_textenc, bert_textenc, device,
    clip_style_text="map in swisstopo style",
    gamma=1.0, w_clip=0.4, bert_align=None, start_idx=8  # start_idx=8 是你打印出来的 [EOS] 后起写位置
):
    """
    取消软混入：让 CFG 差分只来自 BERT（uc 不含 BERT）。
    - ctx = w_clip * clip_ctx + gamma * bert_tail
    - uc  = w_clip * clip_ctx
    """
    mp_dtype = next(model.parameters()).dtype
    seg = seg.to(device=device, dtype=mp_dtype)
    B = seg.size(0)

    # --- 1) CLIP 基座（只编码固定 style 文本）---
    clip_prompts = [clip_style_text] * B
    with torch.no_grad():
        clip_ctx = clip_textenc.get_learned_conditioning(clip_prompts)  # [B,77,768]
    clip_ctx = clip_ctx.to(device=device, dtype=mp_dtype)

    # --- 2) BERT tokens -> [EOS] 后“占位覆盖”为 bert_tail（总长仍 77）---
    #     只 LN/对齐 BERT 路径；CLIP 路径不做 LN
    bert_tok = bert_textenc.encode(prompts).to(device=device, dtype=mp_dtype)  # [B,Lb,768]
    if bert_align is not None:
        bert_tok = bert_align(bert_tok)  # 轻量对齐（0-init + 门控）
    bert_tok = torch.nn.functional.layer_norm(bert_tok, (bert_tok.size(-1),))

    # 写入到 77 长度的尾部区间
    Lb = bert_tok.size(1)
    max_len = clip_ctx.size(1)  # 77
    tail = torch.zeros(B, max_len, clip_ctx.size(-1), device=device, dtype=mp_dtype)
    end = min(max_len, start_idx + Lb)
    if end > start_idx:
        tail[:, start_idx:end, :] = bert_tok[:, :end - start_idx, :]

    # --- 3) 线性融合（激进文本设置）---
    # cond 用 BERT+CLIP，加权；uc 只保留 CLIP（差分由 BERT 提供）
    cond_ctx = w_clip * clip_ctx + gamma * tail
    uc_ctx   = w_clip * clip_ctx

    # --- 4) 组装 ControlNet 条件 ---
    cond = {"c_concat": [seg], "c_crossattn": [cond_ctx]}
    uc   = {"c_concat": [seg], "c_crossattn": [uc_ctx]}
    return cond, uc

# def build_cond_uc_safe_pad(
#     seg, prompts, model, clip_textenc, bert_textenc, device,
#     clip_style_text="map in swisstopo style",
#     gamma=0.25, uc_mix=0.15, bert_align=None
# ):
#     """
#     用“占位覆盖”的方式在 [EOS] 之后塞入 BERT token（总长仍 77）
#     - seg: (B,C,H,W)
#     - prompts: 列表（给 BERT）
#     """
#     mp_dtype = next(model.parameters()).dtype
#     seg = seg.to(device=device, dtype=mp_dtype)
#
#     B = seg.size(0)
#     clip_prompts = [clip_style_text] * B
#     cond_ctx, uc_ctx = fuse_clip_bert_in_pad(
#         clip_textenc=clip_textenc, bert_textenc=bert_textenc,
#         clip_prompts=clip_prompts, bert_prompts=prompts,
#         device=device, gamma=gamma, uc_mix=uc_mix, bert_align=bert_align,
#         max_len=77, dtype=mp_dtype
#     )
#
#     cond = {"c_concat": [seg], "c_crossattn": [cond_ctx]}
#     uc   = {"c_concat": [seg], "c_crossattn": [uc_ctx]}
#     return cond, uc

# 做add tokens的
# def build_cond_uc_safe(
#     seg, prompts, model, clip_textenc, bert_textenc, device,
#     clip_style_text="map in swisstopo style",
#     bert_len_align=None, bert_align=None,
# ):
#     """
#     返回 cond/uc：
#       cond:  c_concat=[seg], c_crossattn=[ concat( CLIP(77), BERT_Adapter(Lb) ) ]
#       uc:    同样保留 CLIP 基底，BERT 部分用全零增量
#     关键点：
#       - c_crossattn 始终是“单元素列表”，内部广播
#       - 不对 CLIP token 做任何归一化/缩放
#     """
#     assert isinstance(prompts, (list, tuple))
#     B = seg.size(0)
#     assert len(prompts) == B, f"len(prompts)={len(prompts)} vs batch={B}"
#     assert seg.dim()==4 and seg.size(1) in (1,3), f"seg shape={tuple(seg.shape)}"
#
#     mp_dtype = next(model.parameters()).dtype
#     seg = seg.to(device=device, dtype=mp_dtype)
#     # print("[debug] seg range:", float(seg.min()), float(seg.max()), float(seg.mean()))
#
#     # === 冻结的 CLIP 基底 ===
#     with torch.no_grad():
#         clip_ctx = clip_textenc.get_learned_conditioning([clip_style_text] * B).to(device=device, dtype=mp_dtype)   # [B,77,768]
#
#     # === BERT 编码（adapter 已内置在 textenc.encode 里）===
#     bert_ctx = bert_textenc.encode(prompts).to(device=device, dtype=mp_dtype)  # [B, Lb, 768]
#     if bert_len_align is not None:
#         bert_ctx = bert_len_align(bert_ctx)  # [B, 77, 768]
#     if bert_align is not None:
#         bert_ctx = bert_align(bert_ctx)
#
#     ctx = clip_ctx + bert_ctx  # 长度仍是 77，避免注意力被稀释
#     ucctx = clip_ctx  # 无条件分支更稳：只用 CLIP 基底
#
#     cond = {"c_concat": [seg], "c_crossattn": [ctx]}
#     uc = {"c_concat": [seg], "c_crossattn": [ucctx]}
#     return cond, uc
# def build_cond_uc_safe(
#     seg, prompts, model, clip_textenc, bert_textenc, device,
#     clip_style_text="map in swisstopo style"
# ):
#     """
#     返回 cond/uc：
#       cond:  c_concat=[seg], c_crossattn=[ concat( CLIP(77), BERT_Adapter(Lb) ) ]
#       uc:    同样保留 CLIP 基底，BERT 部分用全零增量
#     关键点：
#       - c_crossattn 始终是“单元素列表”，内部广播
#       - 不对 CLIP token 做任何归一化/缩放
#     """
#     assert isinstance(prompts, (list, tuple))
#     B = seg.size(0)
#     assert len(prompts) == B, f"len(prompts)={len(prompts)} vs batch={B}"
#     assert seg.dim()==4 and seg.size(1) in (1,3), f"seg shape={tuple(seg.shape)}"
#
#     mp_dtype = next(model.parameters()).dtype
#     seg = seg.to(device=device, dtype=mp_dtype)
#     # print("[debug] seg range:", float(seg.min()), float(seg.max()), float(seg.mean()))
#
#     # 1) CLIP 基底（冻结）— 不做任何归一化/缩放
#     clip_ctx = clip_textenc.get_learned_conditioning([clip_style_text] * len(prompts))
#     clip_ctx = clip_ctx.to(device=device, dtype=mp_dtype)  # (B,77,768)
#
#     # 2) BERT（已内置 Adapter）：直接拿到 (B,Lb,768)
#     bert_ctx = bert_textenc.encode(prompts).to(device=device, dtype=mp_dtype)
#
#     # 3) 沿序列维拼接
#     ctx = torch.cat([clip_ctx, bert_ctx], dim=1)  # (B,77+Lb,768)
#
#     # 4) uc：保留 CLIP 基底，BERT 部分置零
#     bert_empty = bert_textenc.encode([""] * len(prompts)).to(device=device, dtype=mp_dtype)
#     bert_zero = torch.zeros_like(bert_empty)
#     uc_ctx = torch.cat([clip_ctx, bert_zero], dim=1)
#
#     cond = {"c_concat": [seg], "c_crossattn": [ctx]}
#     uc = {"c_concat": [seg], "c_crossattn": [uc_ctx]}
#     return cond, uc

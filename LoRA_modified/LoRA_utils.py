import torch.nn as nn
import torch
import math

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

@torch.no_grad()
def set_lora_condition(root: nn.Module, seg_feat: torch.Tensor, text_tokens: torch.Tensor):
    """
    仅给 LoRALinear 实例注入条件；不会碰到 MemoryEfficientCrossAttention。
    """
    for m in root.modules():
        if isinstance(m, LoRALinear):
            m.set_control_feature(seg_feat, text_tokens)

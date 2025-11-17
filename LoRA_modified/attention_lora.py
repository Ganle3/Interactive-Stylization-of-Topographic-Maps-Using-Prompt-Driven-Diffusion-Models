#  Adapted from HighCWu/ControlLoRA (https://github.com/HighCWu/ControlLoRA)
#  Implemented for ldm.models.diffusion.ddpm.ControlledUnetModel
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from ControlNet.ldm.modules.attention import CrossAttention


# ============================================================
#  1. LoRA 基础层
# ============================================================
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=4, alpha=None):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        in_f, out_f = base.in_features, base.out_features
        self.A = nn.Linear(in_f, r, bias=False)
        self.B = nn.Linear(r, out_f, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.B.weight)
        self.scaling = (alpha or (2 * r)) / r

    def forward(self, x):
        return self.base(x) + self.B(self.A(x)) * self.scaling


# ============================================================
#  2. CrossAttention_LoRA_Control
#      仿 HighCWu: 可接收 control_states 的 CrossAttention
# ============================================================
class CrossAttention_LoRA_Control(CrossAttention):
    def __init__(self, *args, lora_rank=4, control_rank=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.to_q = LoRALinear(self.to_q, r=lora_rank)
        self.to_k = LoRALinear(self.to_k, r=lora_rank)
        self.to_v = LoRALinear(self.to_v, r=lora_rank)
        if isinstance(self.to_out, nn.ModuleList):
            self.to_out[0] = LoRALinear(self.to_out[0], r=lora_rank)
        else:
            self.to_out = nn.ModuleList([LoRALinear(self.to_out, r=lora_rank), nn.Dropout(0.0)])

        # Control 分支: 额外 LoRA 对 control feature 的映射
        control_rank = control_rank or lora_rank
        hidden_size = self.to_q.base.out_features
        self.to_control = nn.Sequential(
            nn.Linear(hidden_size, control_rank, bias=False),
            nn.Linear(control_rank, hidden_size, bias=False)
        )
        nn.init.zeros_(self.to_control[1].weight)
        self.control_states = None

    def inject_control_states(self, control_feat):
        """供外部调用，注入 ControlNet encoder 输出"""
        self.control_states = control_feat

    def process_control_states(self, hidden_states, scale=1.0):
        """仿 HighCWu: 将 control feature 融入 hidden"""
        if self.control_states is None:
            return 0
        control = self.control_states
        if control.ndim == 4 and hidden_states.ndim == 3:
            # ControlNet 输出是 BCHW -> 展平为 B,HW,C
            B, C, H, W = control.shape
            control = control.permute(0, 2, 3, 1).reshape(B, H * W, C)
        control = control.to(hidden_states.dtype)
        return scale * self.to_control(control)

    def forward(self, x, context=None, mask=None, scale=1.0):
        # 主路径
        out = super().forward(x, context=context, mask=mask)
        # 融合 control_states
        if self.control_states is not None:
            ctrl = self.process_control_states(x, scale=scale)
            if ctrl.shape == out.shape:
                out = out + ctrl
        return out


# ============================================================
#  3. ControlLoRA 模块 (生成 control feature maps)
# ============================================================
class ControlLoRA(nn.Module):
    """简化版 ControlNet encoder, 用于产生 control_states"""
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels * 2, 3, 2, 1),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, 2, 1),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.encoder(x)


# ============================================================
#  4. 注入函数：替换 LDM UNet 中的 CrossAttention
# ============================================================
def inject_control_lora_into_ldm(unet, rank=4, control_rank=None):
    count = 0
    for name, module in unet.named_modules():
        if isinstance(module, CrossAttention) and not isinstance(module, CrossAttention_LoRA_Control):
            new_attn = CrossAttention_LoRA_Control(
                query_dim=module.query_dim,
                heads=module.heads,
                dim_head=module.dim_head,
                dropout=module.dropout,
                context_dim=module.context_dim,
                bias=module.to_q.bias is not None,
                lora_rank=rank,
                control_rank=control_rank,
            )
            # 拷贝主干权重
            new_attn.to_q.base.load_state_dict(module.to_q.state_dict())
            new_attn.to_k.base.load_state_dict(module.to_k.state_dict())
            new_attn.to_v.base.load_state_dict(module.to_v.state_dict())
            if isinstance(module.to_out, nn.ModuleList):
                new_attn.to_out[0].base.load_state_dict(module.to_out[0].state_dict())

            # 替换模块
            parent = unet
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], new_attn)
            count += 1

    print(f"[inject_control_lora_into_ldm] Replaced {count} CrossAttention layers with LoRA-Control versions.")
    return count


# ============================================================
#  5. 冻结与优化器辅助
# ============================================================
def freeze_all_but_lora(model):
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, LoRALinear) or hasattr(m, "to_control"):
            for p in m.parameters():
                p.requires_grad = True

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[freeze_all_but_lora] Trainable params: {n_train:,}")
    return n_train


def get_lora_parameters(model):
    return [p for p in model.parameters() if p.requires_grad]
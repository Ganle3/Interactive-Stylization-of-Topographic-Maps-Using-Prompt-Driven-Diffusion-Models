import torch, numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from experiment_overfit.textenc_utils import build_text_encoder  # my encoder structure

# ======================
# 配置路径
# ======================
textenc_ckpt = r"D:\Junyhuang\Project2\Outputs_overfit\Sameseg_multiprompt\wider_adapter_lora12\textenc_adapter_step010000.pt"
device = "cpu"

# ======================
# 载入 text encoder
# ======================
textenc = build_text_encoder(
    n_embed=768, n_layer=12, max_len=77, device=device
).to(device).eval()

state = torch.load(textenc_ckpt, map_location=device)
textenc.load_state_dict(state, strict=False)
print("[✓] loaded text encoder checkpoint")

# ======================
# 定义测试 prompt
# ======================
# prompts = [
#     "Set building color to pink",
#     "Set building color to yellow",
#     "Set building color to blue",
#     "Set building color to brown",
#     "Set building color to red",
#     "keep building original color",
# ]
prompts = [
    "Set Forest color to light green.",
    "Set Forest color to dark green.",
    "Set Building color to gray.",
    "Set Road color to gray.",
    "Set River color to blue.",
    "Keep original color"
]

# ======================
# 计算 embedding
# ======================
with torch.no_grad():
    embeds = []
    for p in prompts:
        e = textenc.encode([p])                 # [1, L, 768]
        pooled = e.mean(dim=1).cpu().numpy()    # 平均池化
        embeds.append(pooled.squeeze())
embeds = np.stack(embeds)

# ======================
# 计算相似度矩阵
# ======================
sim = cosine_similarity(embeds)
print("Cosine similarity matrix:")
print(np.round(sim, 3))

# ======================
# 可视化分布
# ======================
pca = PCA(n_components=2)
xy = pca.fit_transform(embeds)
plt.figure(figsize=(5, 5))
plt.scatter(xy[:, 0], xy[:, 1],
            c=['pink', 'gold', 'skyblue', 'saddlebrown', 'red', 'gray'], s=80)
for i, p in enumerate(prompts):
    plt.text(xy[i, 0] + 0.01, xy[i, 1], p, fontsize=8)
plt.title("Prompt embeddings (BERT+Adapter)")
plt.tight_layout()
plt.savefig("prompt_embedding_pca.png", dpi=200)
print("[viz] saved prompt_embedding_pca.png")

# # ======================
# # 检查 LoRA 状态
# # ======================
# lora_ckpt = r"D:\Junyhuang\Project2\Outputs_overfit\adapter_lora_upscale_bc_of1\best_unet_lora_kv.pt"
# lora_sd = torch.load(lora_ckpt, map_location="cpu")
# print(f"[✓] loaded LoRA weights: {len(lora_sd)} layers")
#
# # 打印几个示例权重的范数
# for k, v in list(lora_sd.items())[:5]:
#     print(f"{k:<80} | norm = {v.norm().item():.4f}")
# ======================
# 检查 LoRA 状态（区分 attn1 / attn2）
# ======================
lora_ckpt = r"D:\Junyhuang\Project2\Outputs_overfit\Sameseg_multiprompt\wider_adapter_lora12\unet_lora_step010000.pt"
lora_sd = torch.load(lora_ckpt, map_location="cpu")
print(f"[✓] loaded LoRA weights: {len(lora_sd)} layers\n")

# 分类统计
attn1_norms, attn2_norms = [], []
attn1_zero, attn2_zero = 0, 0

for k, v in lora_sd.items():
    norm_val = v.norm().item()
    if "attn1" in k:
        attn1_norms.append(norm_val)
        if norm_val < 1e-6:
            attn1_zero += 1
    elif "attn2" in k:
        attn2_norms.append(norm_val)
        if norm_val < 1e-6:
            attn2_zero += 1

# 输出统计
print(f"Total LoRA layers: {len(lora_sd)}")
print(f"  attn1 layers: {len(attn1_norms)}  (zero={attn1_zero})")
print(f"  attn2 layers: {len(attn2_norms)}  (zero={attn2_zero})")
print()

if attn1_norms:
    print(f"attn1.B/A weight norm mean={np.mean(attn1_norms):.6f}, max={np.max(attn1_norms):.6f}")
if attn2_norms:
    print(f"attn2.B/A weight norm mean={np.mean(attn2_norms):.6f}, max={np.max(attn2_norms):.6f}")
print()

# 打印几个具体层作为样例
print("=== Sample attn2 LoRA weights (top 5 by norm) ===")
for k, v in sorted(lora_sd.items(), key=lambda kv: kv[1].norm().item(), reverse=True)[:5]:
    if "attn2" in k:
        print(f"{k:<90} | norm = {v.norm().item():.6f}")

print("\n=== Sample attn1 LoRA weights (top 5 by norm) ===")
for k, v in sorted(lora_sd.items(), key=lambda kv: kv[1].norm().item(), reverse=True)[:5]:
    if "attn1" in k:
        print(f"{k:<90} | norm = {v.norm().item():.6f}")


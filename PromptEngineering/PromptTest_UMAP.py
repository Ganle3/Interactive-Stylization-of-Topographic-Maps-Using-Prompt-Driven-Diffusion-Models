"""
prompt categories:
- Element-specific styling
- Item-specific color control
- Visibility/presence of classes
- Morphology control
- Global tonal adjustments
"""

import torch
from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel
from transformers import T5Tokenizer, T5EncoderModel
from transformers import CLIPTokenizer, CLIPTextModel
import umap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.metrics import pairwise_distances

# Prompt list
# # initial prompt list
# PROMPTS = {
#     "element_styling": [
#         "Render roads in dashed line style",
#         "Render highways with thick marker lines",
#         "Railways in dotted line style",
#         "Rivers with wavy stroke style",
#         "Streams in light dashed lines",
#         "Lakes with bold outline",
#         "Forests in dense fill style",
#         "Trees in cross-hatch shading",
#         "Buildings in marker line style",
#         "Paths in thin dotted lines",
#         "Background with subtle grid pattern"
#     ],
#     "item_color": [
#         "Roads in dark gray",
#         "Highways in bright orange",
#         "Railways in black",
#         "Rivers in bright blue",
#         "Streams in light blue",
#         "Lakes with red outlines",
#         "Forests in dark green",
#         "Trees in light green",
#         "Buildings in yellow",
#         "Settlements in brown",
#         "Background in pale beige"
#     ],
#     "visibility": [
#         "Show only roads",
#         "Hide highways",
#         "Remove railways",
#         "Delete rivers from the map",
#         "Hide all streams",
#         "Remove lakes",
#         "Delete forests from the map",
#         "Hide all trees",
#         "Remove buildings",
#         "Hide settlements",
#         "Remove background grid"
#     ],
#     "morphology": [
#         "Widen the roads",
#         "Narrow the highways",
#         "Thicken the railways",
#         "Increase curvature of rivers",
#         "Smooth the stream lines",
#         "Enlarge lakes",
#         "Round the forest boundaries",
#         "Round the building corners",
#         "Expand settlements area",
#         "Straighten the paths",
#         "Simplify the background contours"
#     ],
#     "global_tonal": [
#         "Dark pastel map",
#         "High-contrast monochrome",
#         "Brighten the entire map",
#         "Reduce map color saturation",
#         "Warm tone adjustment",
#         "Cool tone adjustment",
#         "Sepia style map",
#         "Night mode map",
#         "Soft pastel style",
#         "High saturation map",
#         "Grayscale map"
#     ]
# }

# regularized prompt list
PROMPTS = {
    "element_styling": [
        "Render roads in topo road line style",
        "Render highways in effect neon line style",
        "Render railways in topo railway line style",
        "Render rivers in dashed outline style",
        "Render streams in wavy line style",
        "Render lakes in bold outline style",
        "Render forests in dense fill style",
        "Render trees in star mark style",
        "Render buildings in solid fill style",
        "Render paths in simple line style",
        "Render background in subtle grid style"
    ],
    "item_color": [
        "Set roads color to dark gray",
        "Set highways color to bright orange",
        "Set railways color to black",
        "Set rivers color to bright blue",
        "Set streams color to light blue",
        "Set lake outline color to red",
        "Set forests outline color to dark green",
        "Set trees color to light green",
        "Set building color to yellow",
        "Set building outline color to brown",
        "Set background color to pale beige"
    ],
    # "item_color": [
    #     "Set road color to #444444",       # dark gray
    #     "Set highway color to #FF7F0E",    # bright orange
    #     "Set railway color to #000000",    # black
    #     "Set river color to #1F77B4",      # bright blue
    #     "Set stream color to #AEC7E8",     # light blue
    #     "Set lake outline color to #D62728",    # red
    #     "Set forest outline color to #2CA02C",  # dark green
    #     "Set tree color to #98DF8A",    # light green
    #     "Set building fill color to #FFD700",   # yellow
    #     "Set building outline color to #8B4513",     # brown
    #     "Set background color to #F5F5DC"     # beige
    # ],
    "visibility": [
        "Only make roads visible",
        "Make highways hidden",
        "Make railways hidden",
        "Make rivers hidden",
        "Make streams hidden",
        "Make lakes hidden",
        "Make forests hidden",
        "Make trees hidden",
        "Make buildings hidden",
        "Make buildings outline hidden",
        "Make lakes outline visible"
    ],
    "morphology": [
        "Modify roads to be wider",
        "Modify highways to be narrower",
        "Modify railways to be wider",
        "Modify rivers to be expanded",
        "Modify streams to be smoother",
        "Modify lakes to be larger",
        "Modify forests to have rounded boundaries",
        "Modify buildings to have rounded corners",
        "Modify forests to be expanded",
        "Modify paths to be straighter",
        "Modify background to be simplified"
    ],
    "global_tonal": [
        "Apply dark pastel adjustment to the map",
        "Apply high-contrast monochrome adjustment to the map",
        "Apply brightening adjustment to the map",
        "Apply desaturation adjustment to the map",
        "Apply warm tone adjustment to the map",
        "Apply cool tone adjustment to the map",
        "Apply sepia adjustment to the map",
        "Apply night mode adjustment to the map",
        "Apply soft pastel adjustment to the map",
        "Apply high saturation adjustment to the map",
        "Apply grayscale adjustment to the map"
    ]
}


# 1. initialize text encoder
# BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()
# # RoBERTa
# tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
# model = RobertaModel.from_pretrained("roberta-base")
# model.eval()
# # T5
# tokenizer = T5Tokenizer.from_pretrained("t5-base")
# model = T5EncoderModel.from_pretrained("t5-base")
# model.eval()
# # CLIP
# tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
# model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
# model.eval()

# 2. get embedding
embeddings, labels = [], []
# for CLS vector
for lab, texts in PROMPTS.items():
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        # obtain CLS vector
        cls_emb = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        embeddings.append(cls_emb)
        labels.append(lab)

embeddings = np.vstack(embeddings)

# # for mean-pool
# for lab, texts in PROMPTS.items():
#     for text in texts:
#         inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
#         with torch.no_grad():
#             outputs = model(**inputs)  # last_hidden_state: [1, seq_len, hidden]
#
#         last_hidden = outputs.last_hidden_state           # [1, L, H]
#         mask = inputs["attention_mask"].unsqueeze(-1)     # [1, L, 1]
#         emb = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)  # [1, H]
#         emb = emb.squeeze(0).cpu().numpy()
#
#         embeddings.append(emb)
#         labels.append(lab)
#
# embeddings = np.vstack(embeddings)

# # for pooler_output ([SOS] token embedding)
# for lab, texts in PROMPTS.items():
#     for text in texts:
#         inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
#         with torch.no_grad():
#             outputs = model(**inputs)
#         # pooler_output is [SOS] embedding, shape [1, hidden]
#         emb = outputs.pooler_output.squeeze(0).cpu().numpy()
#         embeddings.append(emb)
#         labels.append(lab)
#
# embeddings = np.vstack(embeddings)

# 3. PCA (optional) + UMAP
X = embeddings  # shape: (n_samples, n_features)
n_samples, n_features = X.shape

pca_dim = min(15, n_samples - 1, n_features)  # Max 15
if pca_dim >= 2:
    X_input = PCA(n_components=pca_dim, random_state=42).fit_transform(X)
else:
    X_input = X

n_neighbors = min(10, max(2, n_samples // 2))

reducer = umap.UMAP(
    n_neighbors=n_neighbors,
    min_dist=0.1,
    metric="cosine",
    random_state=42
)
X_umap = reducer.fit_transform(X_input)

# addition: silhouette score
cats_sorted = sorted(set(labels))
cat2id = {c: i for i, c in enumerate(cats_sorted)}
y = np.array([cat2id[l] for l in labels])
# vectors silhouette（cosine distance）
sil_high = silhouette_score(X, y, metric="cosine")
print(f"[Silhouette @ BERT embeddings (cosine)]: {sil_high:.3f}")
# UMAP 2D coordinates silhouette（Euclidean distance）
sil_umap = silhouette_score(X_umap, y, metric="euclidean")
print(f"[Silhouette @ UMAP 2D (euclidean)]: {sil_umap:.3f}")

# 4. visualization
plt.figure(figsize=(7, 6))
cats = list(dict.fromkeys(labels))
for c in cats:
    idx = [i for i, l in enumerate(labels) if l == c]
    plt.scatter(X_umap[idx,0], X_umap[idx,1], label=c.replace("_", " "), s=70, alpha=0.85)
plt.legend(title="Category")
plt.title("UMAP of Prompt Embeddings (BERT-Base)")
plt.tight_layout()
plt.savefig("umap_BERT_prompts.png", dpi=160)
plt.close()

#  silhouette score of one class with other different categories of prompts
#  every sample's silhouette score (high dim vec + cosine)
sample_sil = silhouette_samples(X, y, metric="cosine")
# calculate according to diff categories
print("\nPer-class silhouette scores with other classes (BERT embeddings, cosine):")
for c in cats_sorted:
    idx = [i for i, lab in enumerate(labels) if lab == c]
    mean_val = sample_sil[idx].mean()
    print(f"  {c:15s}: {mean_val:.3f}")

# analyse one class considering only intra-class compactness
D = pairwise_distances(X, metric="cosine")

print("\nPer-class intra-class compactness (mean pairwise cosine distance):")
for c in cats_sorted:
    idx = [i for i, lab in enumerate(labels) if lab == c]
    subD = D[np.ix_(idx, idx)]
    # dialog = 0
    vals = subD[np.triu_indices_from(subD, k=1)]
    mean_intra = vals.mean() if vals.size > 0 else float("nan")
    print(f"  {c:15s}: {mean_intra:.3f}")

TOPK = 5
TAU  = 0.08   # threshold

# get all original prompts, keep same with labels sequence
all_prompts = [t for cat_list in PROMPTS.values() for t in cat_list]
cats_sorted = sorted(set(labels))

print(f"\n=== Per-class 'most confusable' prompt pairs (cosine distance) ===")
for cat in cats_sorted:
    idx = [i for i, lab in enumerate(labels) if lab == cat]
    subD = D[np.ix_(idx, idx)].copy()

    # delete dialog
    pairs = []
    for a in range(len(idx)):
        for b in range(a+1, len(idx)):
            dist = subD[a, b]
            pairs.append((dist, idx[a], idx[b]))

    if not pairs:
        print(f"\n[{cat}] (n<2) — skip")
        continue

    pairs.sort(key=lambda x: x[0])  # distance smaller, more similar
    print(f"\n[{cat}]  top-{TOPK} closest pairs (smaller = more similar):")
    for k, (dist, i, j) in enumerate(pairs[:TOPK], 1):
        flag = "  <-- near-duplicate" if dist < TAU else ""
        print(f"{k:>2d}. dist={dist:0.3f}{flag}\n    - {all_prompts[i]}\n    - {all_prompts[j]}")

    # refer to the most distinct
    far_dist, fi, fj = max(pairs, key=lambda x: x[0])
    print(f"    farthest pair (dist={far_dist:0.3f}):\n"
          f"    - {all_prompts[fi]}\n    - {all_prompts[fj]}")
# compositionality_test_bert.py
import torch
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from transformers import BertTokenizer, BertModel

# configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIM_MARGIN = 0.05   # similarity threshold as [mean(sim_target)-mean(sim_other)]
TOPK = 8            # categories of nearest prompts

# 1. categorized PROMPTS
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

# 2. combination of prompts
COMPOSED = [
    # color + morphology
    ("Set building fill color to yellow and modify buildings to have rounded corners",
     ["item_color", "morphology"]),
    ("Set road color to dark gray and modify roads to be wider",
     ["item_color", "morphology"]),
    ("Set river color to bright blue and modify streams to be smoother",
     ["item_color", "morphology"]),
    # visibility + style
    ("Make lakes visible to hidden and render rivers in dashed outline style",
     ["visibility", "element_styling"]),
    ("Make forests hidden and render streams in wavy line style",
     ["visibility", "element_styling"]),
    # global_tonal + elements/items
    ("Apply grayscale adjustment to the map and render road in topo road line style",
     ["global_tonal", "element_styling"]),
    ("Apply high saturation adjustment to the map and set forest outline color to dark green",
     ["global_tonal", "item_color"]),
    # triple
    ("Apply warm tone adjustment and set highway color to dark gray and modify roads to be wider",
     ["global_tonal", "item_color", "morphology"]),
]

# 3. initialize BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)
model.eval()

@torch.no_grad()
def encode_one(text: str) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    outputs = model(**inputs)  # last_hidden_state: [1, L, H]
    emb = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()  # CLS
    return emb

# 4. encode single cate prompts, calculate the centroid
embeddings, labels, texts_all = [], [], []
for lab, texts in PROMPTS.items():
    for t in texts:
        e = encode_one(t)
        embeddings.append(e)
        labels.append(lab)
        texts_all.append(t)

X_single = np.vstack(embeddings)      # [N,H]
y_single = np.array(labels)            # [N]
cats = sorted(set(y_single))

centroids = {c: X_single[y_single == c].mean(axis=0) for c in cats}
C_mat = np.vstack([centroids[c] for c in cats])
C_names = cats

print("Built centroids for categories:", C_names)

# 5. check compositionality of every combined prompt
print("\n=== Compositionality sanity check (BERT-CLS) ===")
for text, targets in COMPOSED:
    z = encode_one(text)  # [H]

    # similarity to certain cate's centroid
    sims = cosine_similarity(z.reshape(1, -1), C_mat).ravel()
    sim_table = list(zip(C_names, sims))
    sim_table.sort(key=lambda x: x[1], reverse=True)

    target_sims = [s for (n, s) in sim_table if n in targets]
    other_sims  = [s for (n, s) in sim_table if n not in targets]
    mean_t = float(np.mean(target_sims)) if target_sims else float("nan")
    mean_o = float(np.mean(other_sims))  if other_sims  else float("nan")
    margin = mean_t - mean_o

    # nearest single cate
    dists = cosine_distances(z.reshape(1, -1), X_single).ravel()
    nn_idx = np.argsort(dists)[:TOPK]
    nn_labels = y_single[nn_idx]
    coverage = all(Counter(nn_labels).get(t, 0) > 0 for t in targets)

    # output
    print("\n--------------------------------------------------")
    print(f"COMPOSED: {text}")
    print(f"Targets : {targets}")
    print("Top-5 centroid similarities:")
    for name, s in sim_table[:5]:
        flag = " *" if name in targets else ""
        print(f"  {name:15s}: {s:+.3f}{flag}")
    print(f"mean(target)-mean(other) = {margin:+.3f}  -> {'PASS' if margin >= SIM_MARGIN else 'CHECK'}")
    print(f"NN@{TOPK} coverage of targets: {'OK' if coverage else 'MISSING'}")
    # nearest cates
    # print("NN label counts:", dict(Counter(nn_labels)))

print("\nDone.")

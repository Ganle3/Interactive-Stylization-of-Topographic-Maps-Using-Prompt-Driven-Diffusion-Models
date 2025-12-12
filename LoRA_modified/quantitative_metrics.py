# -*- coding: utf-8 -*-
import os, sys, json, random
import torch
import numpy as np
import torchvision.utils as vutils
from PIL import Image
from math import pi, cos
from torch.utils.data import DataLoader, Subset
sys.path.append(r"D:\Junyhuang\Project2\ControlNet")
sys.path.append(r"D:\Junyhuang\Project2\ctrlora")

# ==== Config ====
ROOTDIR = r"D:\Junyhuang\Project2_Data\Training Data\Item_color"
JSONL   = os.path.join(ROOTDIR, "meta", "pairs.jsonl")
SPLIT_DIR = os.path.join(ROOTDIR, "meta_split")

# ==== CKPT path ====
ADAPTER_CKPT = r"D:\Junyhuang\Project2\Outputs_19prompts_color_ctrlora\textenc_adapter_step086400.pt"
CTRLNET_LORA_CKPT = r"D:\Junyhuang\Project2\Outputs_19prompts_color_ctrlora\ctrlora_ft_step086400.ckpt"

# ==== Output ====
OUTDIR = r"D:\Junyhuang\Project2\Outputs_19prompts_color_ctrlora\Quantitative_Results_ctrlora_step86k"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(os.path.join(OUTDIR, "viz"), exist_ok=True)

# =============== import training utils =================
from TextEncoder_Finetuning.data_utils import stratified_split_ratio, PairsJSONLDataset
from TextEncoder_Finetuning.vis_metrics import save_concept_grid_new, sample_preview

from ctrlora.cldm.model import create_model
from SDFusion_bert.bert_network.network import BERTTextEncoder

from transformers import BertTokenizer, BertModel
from skimage.metrics import peak_signal_noise_ratio as psnr
from lpips import LPIPS

# ============= 1. Load model and apply LoRA =================
device = "cuda" if torch.cuda.is_available() else "cpu"

CTRLORA_CFG = r"D:\Junyhuang\Project2\ctrlora\configs\ctrlora_finetune_sd15_rank12.yaml"
BASE_CKPT   = r"D:\Junyhuang\Project2\BaseModel\Swisstopo.ckpt"

# ---- A) load model skeleton with LoRA structure ----
model = create_model(CTRLORA_CFG).cpu()
sd = torch.load(BASE_CKPT, map_location="cpu")
state = sd.get("state_dict", sd)
missing, unexpected = model.load_state_dict(state, strict=False)
print("[load base] missing:", len(missing), "unexpected:", len(unexpected))
del sd, state
model = model.to(device).eval()

# ---- B) load LoRA finetuned parameters ----
lora_sd = torch.load(CTRLNET_LORA_CKPT, map_location="cpu")
model.control_model.load_state_dict(lora_sd, strict=False)
print("[load lora] loaded from", CTRLNET_LORA_CKPT)

# ============= 2. Load BERT text encoder ====================
def build_text_encoder(n_embed=768, n_layer=12, max_len=77, device="cuda"):
    hf_bert = BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    for p in hf_bert.parameters(): p.requires_grad = False

    class LastStateAdapter(torch.nn.Module):
        def __init__(self, dim=768, hidden=768):
            super().__init__()
            self.ff = torch.nn.Sequential(
                torch.nn.Linear(dim, hidden),
                torch.nn.ReLU(),
                torch.nn.LayerNorm(hidden),
                torch.nn.Linear(hidden, hidden),
                torch.nn.ReLU(),
                torch.nn.LayerNorm(hidden),
                torch.nn.Linear(hidden, dim)
            )
            self.final_ln = torch.nn.LayerNorm(dim)
        def forward(self, x):
            return self.final_ln(self.ff(x))

    adapter = LastStateAdapter().to(device)
    te = BERTTextEncoder(
        n_embed=n_embed,
        n_layer=n_layer,
        max_seq_len=max_len,
        vocab_size=30522,
        device=device,
        use_tokenizer=True
    ).to(device)

    te.hf_bert = hf_bert.to(device)
    te.tokenizer = tokenizer
    te.adapter = adapter
    te.max_len = max_len

    def encode(prompts):
        enc = tokenizer(prompts, return_tensors="pt",
                        padding="max_length", truncation=True,
                        max_length=max_len)
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            out = te.hf_bert(input_ids=ids, attention_mask=mask)
            tokens = out.last_hidden_state
        return te.adapter(tokens)

    te.encode = encode
    return te

textenc = build_text_encoder(device=device)
textenc.load_state_dict(torch.load(ADAPTER_CKPT, map_location=device), strict=False)
textenc = textenc.to(device).eval()
print("[load textenc adapter] loaded from:", ADAPTER_CKPT)

# ============= 3. Prepare test dataset ======================
TEST_JSONL = os.path.join(ROOTDIR, "meta_split", "pairs_test.jsonl")

if not os.path.isfile(TEST_JSONL):
    raise FileNotFoundError(f"test.jsonl not found at {TEST_JSONL}")

print("Using existing test split:", TEST_JSONL)

ds_test = PairsJSONLDataset(TEST_JSONL, ROOTDIR, image_size=512, seg_mode="RGB")

# ----- group by prompt -----
from collections import defaultdict
prompt_map = defaultdict(list)
for i in range(len(ds_test)):
    item = ds_test.items[i]
    prompt = item[0]   # PairsJSONLDataset 保存时 first field is prompt
    prompt_map[prompt].append(i)

prompts = sorted(prompt_map.keys())
print(f"[test] found {len(prompts)} prompts.")

# ----- sample per prompt -----
SAMPLES_PER_PROMPT = 5
selected_indices = []

for p in prompts:
    lst = prompt_map[p]
    if len(lst) <= SAMPLES_PER_PROMPT:
        selected_indices.extend(lst)
    else:
        selected_indices.extend(random.sample(lst, SAMPLES_PER_PROMPT))

print(f"[test] selected total {len(selected_indices)} samples")

# For visualization: 19 prompts → 19 samples + extra 1 = 20
vis_indices = []
for p in prompts:
    vis_indices.append(prompt_map[p][0])
# extra 1 from last prompt
if len(prompt_map[prompts[-1]]) > 1:
    vis_indices.append(prompt_map[prompts[-1]][1])

dl_test = DataLoader(
    Subset(ds_test, selected_indices),
    batch_size=1, shuffle=False, num_workers=0
)


# ============= 4. Metric functions ==========================
lpips_fn = LPIPS(net="vgg").to(device)

def mse_rgb(pred, gt):
    return torch.mean((pred - gt) ** 2).item()

def psnr_rgb(pred, gt):
    pred_np = pred.cpu().numpy()
    gt_np   = gt.cpu().numpy()
    return psnr(gt_np, pred_np, data_range=255)

def lpips_rgb(pred, gt):
    # convert to [-1,1]
    pred_n = (pred/127.5 - 1).to(device)
    gt_n   = (gt/127.5 - 1).to(device)
    with torch.no_grad():
        return lpips_fn(pred_n, gt_n).item()

def color_histogram_l1_255(pred, gt, bins=32):
    """
    pred, gt: torch tenser shape [3,H,W], value in 0-255
    Computes L1 distance between RGB histograms in 0-255 space.
    """

    # [3,H,W] → [H,W,3]
    pred = pred.permute(1, 2, 0).cpu().numpy()
    gt = gt.permute(1, 2, 0).cpu().numpy()

    total = 0.0
    for ch in range(3):
        hist_p, _ = np.histogram(pred[:,:,ch], bins=bins, range=(0,255), density=True)
        hist_g, _ = np.histogram(gt[:,:,ch], bins=bins, range=(0,255), density=True)
        total += np.abs(hist_p - hist_g).sum()

    return total / 3.0


# ============= 5. Test Loop ================================
import csv
csv_path = os.path.join(OUTDIR, "metrics.csv")
csv_file = open(csv_path, "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["idx", "prompt", "MSE", "PSNR", "LPIPS"])

global_mse, global_psnr, global_lpips, global_histdis = [], [], [], []

print("\n===== Start Test =====")
with torch.no_grad():
    for bi, batch in enumerate(dl_test):
        seg = batch["seg"].to(device)
        gt  = batch["gt"].to(device)

        pred = sample_preview(
            model, textenc,
            batch, device,
            steps=12, scale=7.5, eta=0.0, seed=1234
        )
        pred = pred.clamp(-1,1)

        # convert pred/gt to RGB 0-255
        pred255 = ((pred + 1) * 127.5).clamp(0,255).round().squeeze(0).cpu()
        gt255   = ((gt   + 1) * 127.5).clamp(0,255).round().squeeze(0).cpu()

        # metrics
        mse_v  = mse_rgb(pred255, gt255)
        psnr_v = psnr_rgb(pred255, gt255)
        lpips_v = lpips_rgb(pred255, gt255)
        hist_diff = color_histogram_l1_255(pred255, gt255)

        prompt = batch["prompt"][0]

        global_mse.append(mse_v)
        global_psnr.append(psnr_v)
        global_lpips.append(lpips_v)
        global_histdis.append(hist_diff)

        writer.writerow([bi, prompt, mse_v, psnr_v, lpips_v, hist_diff])

csv_file.close()

print("\n===== Metrics Summary =====")
print("MSE :", np.mean(global_mse))
print("PSNR:", np.mean(global_psnr))
print("LPIPS:", np.mean(global_lpips))
print("Histogram difference:", np.mean(global_histdis))


# ============= 6. Visualization (20 samples) ================
vis_ds = DataLoader(Subset(ds_test, vis_indices), batch_size=1, shuffle=False)

groups = []
group = []
for i, batch in enumerate(vis_ds):
    group.append(batch)
    if len(group) == 5:
        groups.append(group)
        group = []
# last group automatically included

for gi, grp in enumerate(groups):
    seg_list, pred_list, gt_list, prompts_list = [], [], [], []
    for batch in grp:
        seg     = batch["seg"].to(device)
        gt      = batch["gt"].to(device)
        prompt  = batch["prompt"][0]

        pred = sample_preview(
            model, textenc,
            batch, device,
            steps=12, scale=7.5, eta=0.0, seed=1234
        ).clamp(-1,1)

        seg_list.append(seg)
        pred_list.append(pred)
        gt_list.append(gt)
        prompts_list.append(prompt)

    out_png = os.path.join(OUTDIR, f"viz/group_{gi:02d}.png")
    save_concept_grid_new(seg_list, pred_list, gt_list, prompts_list, out_png)

print("==== Test Done ====")

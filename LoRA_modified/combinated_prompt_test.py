# -*- coding: utf-8 -*-
import os, sys, json, random
import torch
import numpy as np
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torchvision.utils import save_image
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
ADAPTER_CKPT = r"D:\Junyhuang\Project2\Outputs_20prompts_styling_32ctrl_8unt\textenc_adapter_step150000.pt"
CTRLNET_LORA_CKPT = r"D:\Junyhuang\Project2\Outputs_20prompts_styling_32ctrl_8unt\ctrlora_ft_step150000.ckpt"
UNET_LORA_CKPT = r"D:\Junyhuang\Project2\Outputs_20prompts_styling_32ctrl_8unt\unet_lora_step150000.ckpt"

# ==== Output ====
OUTDIR = r"D:\Junyhuang\Project2\Outputs_20prompts_styling_32ctrl_8unt\generate_test_step100k"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(os.path.join(OUTDIR, "viz"), exist_ok=True)

# =============== import training utils =================
from TextEncoder_Finetuning.data_utils import stratified_split_ratio, PairsJSONLDataset
from TextEncoder_Finetuning.vis_metrics import save_concept_grid_new, sample_preview

from ctrlora.cldm.model import create_model
from SDFusion_bert.bert_network.network import BERTTextEncoder
from LoRA_utils import LoRALinear, lora_qkv

from transformers import BertTokenizer, BertModel
from skimage.metrics import peak_signal_noise_ratio as psnr
from lpips import LPIPS

# ============= 1. Load model and apply LoRA =================
device = "cuda" if torch.cuda.is_available() else "cpu"

CTRLORA_CFG = r"D:\Junyhuang\Project2\ctrlora\configs\ctrlora_finetune_sd15_rank32.yaml"
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

# ---- C) load LoRA in main U-Net ----
wrapped = lora_qkv(model.model.diffusion_model, r_q=8, r_kv=8)
print("[unet] injected LoRA structure: Q={}, K={}, V={}".format(
    wrapped["q"], wrapped["k"], wrapped["v"]
))
unet_lora_sd = torch.load(UNET_LORA_CKPT, map_location="cpu")
missing_u, unexpected_u = model.model.diffusion_model.load_state_dict(
    unet_lora_sd, strict=False
)
print(f"[unet lora loaded] missing={len(missing_u)}, unexpected={len(unexpected_u)}")


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

                torch.nn.Linear(hidden, hidden),
                torch.nn.ReLU(),
                torch.nn.LayerNorm(hidden),

                torch.nn.Linear(hidden, hidden),
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


# test seg and prompt
seg_paths = [
    r"D:\Junyhuang\Project2_Data\Training Data\PromptCate_SameSegs\Item_Color\source\tree_darkgreen_004441.png",
    r"D:\Junyhuang\Project2_Data\Training Data\PromptCate_SameSegs\Item_Color\source\tree_darkgreen_004441.png",
    r"D:\Junyhuang\Project2_Data\Training Data\PromptCate_SameSegs\Item_Color\source\tree_darkgreen_004441.png",
    r"D:\Junyhuang\Project2_Data\Training Data\PromptCate_SameSegs\Item_Color\source\tree_darkgreen_004441.png",
    r"D:\Junyhuang\Project2_Data\Training Data\PromptCate_SameSegs\Item_Color\source\tree_darkgreen_004441.png",
    r"D:\Junyhuang\Project2_Data\Training Data\PromptCate_SameSegs\Item_Color\source\tree_darkgreen_004441.png",
    # r"D:\Junyhuang\Project2_Data\Training Data\Item_color_Copy\source\building_pink_000510.png",
    # r"D:\Junyhuang\Project2_Data\Training Data\Item_color_Copy\source\building_pink_000562.png",
    # r"D:\Junyhuang\Project2_Data\Training Data\Item_color_Copy\source\building_pink_000562.png",
]

prompts = [
    "Render Forest with a diagonal hatch fill texture.",
    "Render Tree as a small triangle-shaped mark symbol.",
    "Render Road as a dashed line pattern. ",
    "Keep all elements visible.",
    "Render Tree as a small triangle-shaped mark symbol, Render Forest with a diagonal hatch fill texture.",
    "Render Road as a dashed line pattern, Render Tree as a small triangle-shaped mark symbol",
]

assert len(seg_paths) == len(prompts), "数量不匹配！seg_paths 和 prompts 必须一一对应"


transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
])

def load_seg(p):
    return transform(Image.open(p).convert("RGB")).unsqueeze(0).to(device)

segs = [load_seg(p) for p in seg_paths]

# ---------------- Inference ----------------
print("Start inference...")

def vis_concat(seg, pred):
    """拼接成 1×2 横向图：左 seg, 右 pred"""
    seg_vis  = seg
    pred_vis = (pred * 0.5 + 0.5).clamp(0,1)
    return torch.cat([seg_vis, pred_vis], dim=3)

for i, (seg, prompt) in enumerate(zip(segs, prompts)):
    tokens = textenc.encode([prompt]).to(device)

    cond = {
        "c_crossattn": [tokens],
        "c_concat":    [seg],
    }

    with torch.no_grad():
        samples, _ = model.sample_log(
            cond=cond,
            batch_size=1,
            ddim=True,
            ddim_steps=15,
            eta=0.0,
        )
        x_sample = model.decode_first_stage(samples).clamp(-1,1)

    # 拼接可视化
    merged = vis_concat(seg, x_sample)

    out_path = os.path.join(
        OUTDIR,
        f"infer_{i:02d}_{prompt.replace(' ', '_').replace('.', '')}.png"
    )
    save_image(merged, out_path)

    print(f"[save] {out_path}")
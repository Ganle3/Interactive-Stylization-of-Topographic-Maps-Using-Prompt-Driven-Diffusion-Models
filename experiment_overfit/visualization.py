# -*- coding: utf-8 -*-
import os, torch, json, sys, math
sys.path.append(r"D:\Junyhuang\Project2\ControlNet")

from omegaconf import OmegaConf
from ControlNet.ldm.util import instantiate_from_config
from transformers import BertModel, BertTokenizer
from TextEncoder_Finetuning.data_utils import PairsJSONLDataset
from TextEncoder_Finetuning.vis_metrics import sample_preview, save_concept_grid_new
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F

# ==========================================================
# 路径配置
# ==========================================================
CKPT = r"D:\Junyhuang\Project2\BaseModel\Swisstopo.ckpt"
CFG  = r"D:\Junyhuang\Project2\ControlNet\models\cldm_v15.yaml"

OUTDIR = r"D:\Junyhuang\Project2\Outputs_overfit\pilot_adapter2_lora16_lcc_of"
ROOTDIR = r"D:\Junyhuang\Project2_Data\Training Data\Pilot_color"
JSONL = os.path.join(ROOTDIR, "meta", "pairs.jsonl")

ADAPTER_PT = os.path.join(OUTDIR, "textenc_adapter_step001000.pt")
LORA_PT = os.path.join(OUTDIR, "unet_lora_step001000.pt")
SAVE_DIR = os.path.join(OUTDIR, "revisualize_step1000")
os.makedirs(SAVE_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================================
# Build LoRA wrapper (same as training)
# ==========================================================
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=16, alpha=None):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        in_f, out_f = base.in_features, base.out_features
        dev = base.weight.device
        dty = base.weight.dtype
        self.A = nn.Linear(in_f, r, bias=False).to(device=dev, dtype=dty)
        self.B = nn.Linear(r, out_f, bias=False).to(device=dev, dtype=dty)
        nn.init.zeros_(self.B.weight)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        self.scaling = (alpha or (2*r)) / r

    def forward(self, x):
        return self.base(x) + self.B(self.A(x)) * self.scaling

def lora_kv(unet, r=16):
    n = 0
    for m in unet.modules():
        if hasattr(m, "to_k") and isinstance(m.to_k, nn.Linear):
            m.to_k = LoRALinear(m.to_k, r=r); n += 1
        if hasattr(m, "to_v") and isinstance(m.to_v, nn.Linear):
            m.to_v = LoRALinear(m.to_v, r=r); n += 1
    return n

# ==========================================================
# Build text encoder (same adapter as training)
# ==========================================================
class LastStateAdapter(nn.Module):
    def __init__(self, dim=768, hidden=768):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, dim, bias=True),
        )
        self.final_ln = nn.LayerNorm(dim)
    def forward(self, x):
        y = self.ff(x)
        y = self.final_ln(y)
        return y

from SDFusion_bert.bert_network.network import BERTTextEncoder

def build_text_encoder(device="cuda", max_len=77):
    hf_bert = BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    hf_bert.to(device)
    for p in hf_bert.parameters():
        p.requires_grad = False
    adapter = LastStateAdapter(768).to(device)
    te = BERTTextEncoder(
        n_embed=768, n_layer=12, max_seq_len=max_len,
        vocab_size=30522, device=device, use_tokenizer=True, embedding_dropout=0.0
    ).to(device)
    te.hf_bert = hf_bert
    te.tokenizer = tokenizer
    te.adapter = adapter
    def encode(prompts):
        enc = tokenizer(prompts, return_tensors='pt', padding="max_length", truncation=True, max_length=max_len)
        input_ids = enc['input_ids'].to(device)
        attn_mask = enc['attention_mask'].to(device)
        with torch.no_grad():
            out = hf_bert(input_ids=input_ids, attention_mask=attn_mask, return_dict=True)
            tokens = out.last_hidden_state
        tokens = adapter(tokens)
        return tokens
    te.encode = encode
    return te

# ==========================================================
# Load base model and LoRA/adapter weights
# ==========================================================
print("[load] Base Swisstopo + LoRA/Adapter weights...")

cfg = OmegaConf.load(CFG)
model = instantiate_from_config(cfg.model)
sd = torch.load(CKPT, map_location="cpu")
sd = sd.get("state_dict", sd)
model.load_state_dict(sd, strict=False)
model.to(device).eval()

wrapped = lora_kv(model.model.diffusion_model, r=16)
lora_sd = torch.load(LORA_PT, map_location=device)
model.model.diffusion_model.load_state_dict(lora_sd, strict=False)
print(f"Loaded LoRA weights: {len(lora_sd)} tensors")

textenc = build_text_encoder(device=device)
textenc.load_state_dict(torch.load(ADAPTER_PT, map_location=device), strict=False)
textenc.eval()
print("Loaded adapter weights successfully.")

# ==========================================================
# Build val loader
# ==========================================================
ds = PairsJSONLDataset(JSONL, ROOTDIR, image_size=512, seg_mode="RGB")
ds_val = Subset(ds, [8, 208, 408, 608, 808, 1008])
dl_val = DataLoader(ds_val, batch_size=6, shuffle=False, num_workers=0)

# ==========================================================
# Visualize predictions
# ==========================================================
with torch.no_grad():
    for vb in dl_val:
        seg_v = vb["seg"].to(device, torch.float32)
        gts_v = vb["gt"].to(device, torch.float32)
        proms = vb["prompt"]
        K = gts_v.size(0)
        segs_list, preds_list, gts_list = [], [], []
        for i in range(K):
            seg_i = seg_v[i:i+1]
            gt_i  = gts_v[i:i+1]
            p_i   = [proms[i]]
            pred_i = sample_preview(
                model, textenc,
                {"seg": seg_i, "gt": gt_i, "prompt": p_i},
                device, steps=20, scale=7.5, eta=0.0, seed=1234
            )
            segs_list.append(seg_i)
            preds_list.append(pred_i)
            gts_list.append(gt_i)
        out_png = os.path.join(SAVE_DIR, "revisualize_grid_step1000.png")
        save_concept_grid_new(segs_list, preds_list, gts_list, proms, out_png)
        print(f"[saved] {out_png}")
        break

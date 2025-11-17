import os, math, json, sys
sys.path.append(r"D:\Junyhuang\Project2\ControlNet")
sys.path.append(r"D:\Junyhuang\Project2\ctrlora")
from PIL import Image
import torch
from torchvision import transforms
from omegaconf import OmegaConf
from ControlNet.ldm.util import instantiate_from_config

from ctrlora.cldm.model import create_model

CTRLORA_CFG = r"D:\Junyhuang\Project2\ctrlora\configs\ctrlora_finetune_sd15_rank12.yaml"
BASE_CKPT   = r"D:\Junyhuang\Project2\BaseModel\Swisstopo.ckpt"
LORA_CKPT   = r"D:\Junyhuang\Project2\Outputs_overfit\Sameseg_multiprompt\batch_switch_ctrlora\ctrlora_lora_step008000.ckpt"
TEXTENC_CKPT= r"D:\Junyhuang\Project2\Outputs_overfit\Sameseg_multiprompt\batch_switch_ctrlora\textenc_adapter_step008000.pt"
OUTDIR  = r"D:\Junyhuang\Project2\Outputs_overfit\Sameseg_multiprompt\batch_switch_ctrlora\gen_test_8000"
os.makedirs(OUTDIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# === 1. 构建 UNet + LoRA ===
model = create_model(CTRLORA_CFG).cpu()
# --- 加载完整 SD+ControlNet backbone ---
sd = torch.load(BASE_CKPT, map_location="cpu")
state_dict = sd.get("state_dict", sd)
model.load_state_dict(state_dict, strict=False)

lora_sd = torch.load(LORA_CKPT, map_location="cpu")
missing, unexpected = model.control_model.load_state_dict(lora_sd, strict=False)
print(f"[load] ctrlora LoRA: missing={len(missing)}, unexpected={len(unexpected)}")

model = model.to(device).eval()

# === 2. Load your trained BERT + adapter ===
from transformers import BertModel, BertTokenizer
from SDFusion_bert.bert_network.network import BERTTextEncoder

MAXLEN = 77

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
hf_bert = BertModel.from_pretrained("bert-base-uncased").to(device).eval()

# 构建与训练时完全一致的 BERTTextEncoder 结构
from experiment_overfit.textenc_utils import LastStateAdapter_ctrlora
adapter = LastStateAdapter_ctrlora(dim=768, hidden=768).to(device)

textenc = BERTTextEncoder(
    n_embed=768, n_layer=12, max_seq_len=MAXLEN,
    vocab_size=30522, device=device
).to(device)

textenc.hf_bert = hf_bert
textenc.tokenizer = tokenizer
textenc.adapter = adapter

# --- 加载你训练 8000 步的权重 ---
textenc.load_state_dict(torch.load(TEXTENC_CKPT, map_location=device), strict=False)
textenc.to(device).eval()

def encode_prompt(prompts):
    enc = tokenizer(prompts, return_tensors='pt',
                    padding="max_length", truncation=True, max_length=MAXLEN).to(device)
    with torch.no_grad():
        out = hf_bert(**enc, return_dict=True)
        tokens = out.last_hidden_state
    tokens = adapter(tokens)
    return tokens


# === 3. 加载要测试的seg图像 ===
seg_paths = [
    r"D:\Junyhuang\Project2_Data\Training Data\Sameseg_diffitem_8\source\group_00007_seg.png",
    r"D:\Junyhuang\Project2_Data\Training Data\Item_color\source\building_pink_000510.png",
    r"D:\Junyhuang\Project2_Data\Training Data\Item_color\source\building_pink_000510.png",
    r"D:\Junyhuang\Project2_Data\Training Data\Item_color\source\building_pink_000562.png",
    r"D:\Junyhuang\Project2_Data\Training Data\Item_color\source\building_pink_000562.png",
    r"D:\Junyhuang\Project2_Data\Training Data\Item_color\source\building_pink_000562.png",
    r"D:\Junyhuang\Project2_Data\Training Data\Item_color\source\building_pink_000510.png",
    r"D:\Junyhuang\Project2_Data\Training Data\Item_color\source\building_pink_000510.png",
    r"D:\Junyhuang\Project2_Data\Training Data\Item_color\source\building_pink_000510.png",
    r"D:\Junyhuang\Project2_Data\Training Data\Item_color\source\building_pink_000510.png",
]
transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor()
])

segs = [transform(Image.open(p).convert("RGB")).unsqueeze(0).to(device) for p in seg_paths]

# === 4. 自定义 prompt 测试列表 ===
prompts = [
    "Set Building color to gray.",
    "Set Building color to gray.",
    "Set Road color to gray.",
    "Set River color to light blue.",
    "Set River color to dark blue.",
    "Keep original color.",
    "Set Forest color to light green.",
    "Set Forest color to dark green.",
    "Set Road color to light green.",
    "Keep original color"
]

# === 5. 推理生成 ===
from torchvision.utils import save_image
model.eval()

for i, (seg, prompt) in enumerate(zip(segs, prompts)):
    tokens = encode_prompt([prompt]).to(device=device, dtype=torch.float32)

    cond = {"c_crossattn": [tokens], "c_concat": [seg]}

    with torch.no_grad():
        out = model.sample_log(cond, batch_size=1, ddim=True, ddim_steps=30, eta=0.0)
        samples = out[0] if isinstance(out, (list, tuple)) else out  # 取出真正tensor
        x_sample = model.decode_first_stage(samples)

    out_path = os.path.join(OUTDIR, f"test_{i:02d}_{prompt.replace(' ', '_')}.png")
    save_image((x_sample.clamp(-1,1) + 1) / 2, out_path)
    print(f"[save] {out_path}")
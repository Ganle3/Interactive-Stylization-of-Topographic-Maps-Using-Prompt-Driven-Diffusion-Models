import os, math, json, sys
sys.path.append(r"D:\Junyhuang\Project2\ControlNet")
from PIL import Image
import torch
from torchvision import transforms
from omegaconf import OmegaConf
from ControlNet.ldm.util import instantiate_from_config

# === 基本配置 ===
CKPT    = r"D:\Junyhuang\Project2\BaseModel\Swisstopo.ckpt"
CFG     = r"D:\Junyhuang\Project2\ControlNet\models\cldm_v15.yaml"
OUTDIR  = r"D:\Junyhuang\Project2\Outputs_overfit\Sameseg_multiprompt\batch_switch_qkv\gen_test"
os.makedirs(OUTDIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# === 1. 构建 UNet + LoRA ===
cfg = OmegaConf.load(CFG)
model = instantiate_from_config(cfg.model)
sd = torch.load(CKPT, map_location="cpu"); sd = sd.get("state_dict", sd)
model.load_state_dict(sd, strict=False)
model = model.to(device).eval()

# --- 注入 LoRA (与训练保持一致) ---
from textenc_utils import LoRALinear, lora_qkv, ensure_lora_to_device  # 复用原脚本里的函数
wrapped = lora_qkv(model.model.diffusion_model, r_q=4, r_kv=12)
print("LoRA-wrapped linears:", wrapped)

# --- 加载15000步的LoRA参数 ---
# === 加载 Q/K/V 的 LoRA 权重 ===
LORA_Q_PT  = r"D:\Junyhuang\Project2\Outputs_overfit\Sameseg_multiprompt\batch_switch_qkv\unet_loraQ_step016000.pt"
LORA_KV_PT = r"D:\Junyhuang\Project2\Outputs_overfit\Sameseg_multiprompt\batch_switch_qkv\unet_loraKV_step016000.pt"

# 加载 Q-LoRA
if os.path.exists(LORA_Q_PT):
    lora_q_sd = torch.load(LORA_Q_PT, map_location=device)
    missing, unexpected = model.model.diffusion_model.load_state_dict(lora_q_sd, strict=False)
    print(f"[load] Q-LoRA weights loaded ✓ missing={len(missing)}, unexpected={len(unexpected)}")
else:
    print(f"[warn] Q-LoRA file not found: {LORA_Q_PT}")

# 加载 K/V-LoRA
if os.path.exists(LORA_KV_PT):
    lora_kv_sd = torch.load(LORA_KV_PT, map_location=device)
    missing, unexpected = model.model.diffusion_model.load_state_dict(lora_kv_sd, strict=False)
    print(f"[load] KV-LoRA weights loaded ✓ missing={len(missing)}, unexpected={len(unexpected)}")
else:
    print(f"[warn] KV-LoRA file not found: {LORA_KV_PT}")

# 确保 LoRA 模块在正确的设备上
ensure_lora_to_device(model.model.diffusion_model, device)

# === 2. 构建 TextEncoder + Adapter ===
from transformers import BertModel, BertTokenizer
from SDFusion_bert.bert_network.network import BERTTextEncoder

MAXLEN = 77
hf_bert = BertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
hf_bert.to(device).eval()

# === 2. 构建 TextEncoder + Adapter ===
from transformers import BertModel, BertTokenizer
from SDFusion_bert.bert_network.network import BERTTextEncoder
from textenc_utils import LastStateAdapter  # 引入定义的 adapter 类

hf_bert = BertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
adapter = LastStateAdapter(dim=768, hidden=1536)

textenc = BERTTextEncoder(
    n_embed=768, n_layer=12, max_seq_len=77,
    vocab_size=30522, device=device, embedding_dropout=0.0
).to(device)
textenc.hf_bert = hf_bert
textenc.tokenizer = tokenizer
textenc.adapter = adapter

# 然后直接加载整个权重
ckpt_path = r"D:\Junyhuang\Project2\Outputs_overfit\Sameseg_multiprompt\batch_switch_qkv\textenc_adapter_step016000.pt"
print(f"[load] loading full text encoder weights from {ckpt_path}")
textenc.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
print("[load] text encoder weights loaded ✓")

textenc.to(device)
textenc.hf_bert.to(device)
textenc.adapter.to(device)

def encode_prompt(prompts):
    enc = tokenizer(prompts, return_tensors='pt',
                    padding="max_length", truncation=True,
                    max_length=MAXLEN).to(device)
    with torch.no_grad():
        out = hf_bert(**enc, return_dict=True)
        tokens = out.last_hidden_state
    tokens = adapter(tokens)
    return tokens

# === 3. 加载要测试的seg图像 ===
seg_paths = [
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
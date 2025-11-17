# introduce SDFusion text encoder
from SDFusion_bert.bert_network.network import BERTTextEncoder
from transformers import BertModel, BertTokenizer
hf_bert = BertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ======= debug_ffn_structure =======
import sys, os, torch
sys.path.append(r"D:\Junyhuang\Project2\ControlNet")
from omegaconf import OmegaConf
from ControlNet.ldm.util import instantiate_from_config

CKPT = r"D:\Junyhuang\Project2\BaseModel\Swisstopo.ckpt"
CFG  = r"D:\Junyhuang\Project2\ControlNet\models\cldm_v15.yaml"

cfg = OmegaConf.load(CFG)
model = instantiate_from_config(cfg.model)
sd = torch.load(CKPT, map_location="cpu")
sd = sd.get("state_dict", sd)
model.load_state_dict(sd, strict=False)
unet = model.model.diffusion_model  # <-- our target module

# ---- 选一个 Transformer block 看结构 ----
block = unet.output_blocks[11][1].transformer_blocks[0]
print(block)
print("\n--- FeedForward details ---")
print(block.ff)
print("\n--- FeedForward internal net ---")
print(block.ff.net)

# 可以用这个看看内部结构类型
for i, sub in enumerate(block.ff.net):
    print(f"  [{i}] type={type(sub)}")
    if hasattr(sub, 'proj'):
        print(f"     has proj: {sub.proj}")

# freeze all parameters and only release the last layer
for p in hf_bert.parameters():
    p.requires_grad = False
for p in hf_bert.encoder.layer[-1].parameters():
    p.requires_grad = True
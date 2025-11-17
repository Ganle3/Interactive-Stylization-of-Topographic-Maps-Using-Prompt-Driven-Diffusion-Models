# -*- coding: utf-8 -*-
import os, math, json, sys
sys.path.append(r"D:\Junyhuang\Project2\ControlNet")
import matplotlib.pyplot as plt

# hard-disable the checkpoint wrapper everywhere
def _no_checkpoint(func, inputs, params=None, flag=None):
    # Run forward as-is, no recomputation checkpointing
    return func(*inputs)

# Patch util.checkpoint
import importlib
ldm_util = importlib.import_module("ldm.modules.diffusionmodules.util")
ldm_util.checkpoint = _no_checkpoint

# Patch the *name* 'checkpoint' inside attention.py as well (it does: from util import checkpoint)
ldm_attn = importlib.import_module("ldm.modules.attention")
ldm_attn.checkpoint = _no_checkpoint

print("[patch] Disabled ldm checkpoint: util.checkpoint AND attention.checkpoint are now no-ops.")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from ControlNet.ldm.util import instantiate_from_config
from torch.amp import GradScaler, autocast
from transformers import BertModel, BertTokenizer
import random, numpy as np

import ControlNet.ldm.modules.diffusionmodules.util as ldm_util
from SDFusion_bert.bert_network.network import BERTTextEncoder


def disable_bert_dropout(hf_bert):
    """将 HuggingFace BERT 模型的所有 dropout 层关闭。"""
    # 修改 config 层面的概率
    if hasattr(hf_bert, "config"):
        hf_bert.config.hidden_dropout_prob = 0.0
        hf_bert.config.attention_probs_dropout_prob = 0.0

    # 修改模块实例里的 Dropout 层
    import torch.nn as nn
    for m in hf_bert.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0


def build_text_encoder(n_embed=768, n_layer=12, max_len=77, device="cuda", use_tokenizer=True):
    """
    Build SDFusion's BERTTextEncoder, return tokens [B, L, n_embed]。
    Default to froze layers and only train a adapter.
      tokens = Adapter( last_hidden_state )
    """
    print("[Init] Loading Hugging Face pretrained weights from 'bert-base-uncased'...")

    hf_bert = BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    hf_bert.to(device)

    # froze all parameters from hugging face bert textencoder
    for p in hf_bert.parameters():
        p.requires_grad = False

    # # build a small adapter：LN -> Linear(768->768, initial zeros) -> sigmoid controller -> residual to input
    # class LastStateAdapter(nn.Module):
    #     def __init__(self, dim=768, gate_init=-0.5):
    #         super().__init__()
    #         self.ln = nn.LayerNorm(dim)
    #         self.proj = nn.Linear(dim, dim, bias=True)
    #         nn.init.zeros_(self.proj.weight)
    #         nn.init.zeros_(self.proj.bias)
    #         self.beta = nn.Parameter(torch.tensor(gate_init))  # sigmoid(-0.5)≈0.38, upscale already (small start, no learn
    #     def forward(self, x):  # x: (B, L, 768)
    #         y = self.proj(self.ln(x))
    #         y = torch.sigmoid(self.beta) * y
    #         return x + y  # smaller residual
    # LN → Linear → GELU → Linear
    class LastStateAdapter(nn.Module):
        def __init__(self, dim=768, hidden=1536):
            super().__init__()

            self.ff = nn.Sequential(
                nn.Linear(dim, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),

                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.LayerNorm(hidden),

                nn.Linear(hidden, dim, bias=True),
            )
            self.final_ln = nn.LayerNorm(dim) # mimics CLIP's final LayerNorm

        def forward(self, x):
            y = self.ff(x)
            y = self.final_ln(y)
            return y

    adapter = LastStateAdapter(dim=n_embed).to(device)

    te = BERTTextEncoder(
        n_embed=n_embed, n_layer=n_layer, max_seq_len=max_len,
        vocab_size=30522, device=device, use_tokenizer=use_tokenizer, embedding_dropout=0.0
    ).to(device)

    # add adapter
    te.hf_bert = hf_bert
    te.tokenizer = tokenizer
    te.max_len = max_len
    te.adapter = adapter

    # return Adapter(last_hidden_state)
    def encode_on_device(prompts):
        enc = tokenizer(
            prompts, return_tensors='pt',
            padding="max_length", truncation=True, max_length=max_len
        )
        input_ids = enc['input_ids'].to(device)
        attn_mask = enc['attention_mask'].to(device)
        te.hf_bert.eval()
        with torch.no_grad():
            out = te.hf_bert(input_ids=input_ids, attention_mask=attn_mask, return_dict=True)
            tokens = out.last_hidden_state  # (B,77,768)
        tokens = te.adapter(tokens)        # only layer to train
        return tokens

    te.encode = encode_on_device

    trainable = sum(p.numel() for p in te.adapter.parameters() if p.requires_grad)
    print(f"[Init] BERTTextEncoder ready. Trainable adapter params: {trainable}")
    return te

# Adding LoRA to UNet cross-attn k/v
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=16, alpha=None):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        in_f, out_f = base.in_features, base.out_features
        dev = base.weight.device
        dty = base.weight.dtype

        # A/B create on the same device & dtype with base
        self.A = nn.Linear(in_f, r, bias=False).to(device=dev, dtype=dty)
        self.B = nn.Linear(r, out_f, bias=False).to(device=dev, dtype=dty)

        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

        self.scaling = (alpha or (2*r)) / r

    def forward(self, x):
        # x and base/A/B in same device/dtype
        return self.base(x) + self.B(self.A(x)) * self.scaling


def lora_kv(unet, r=16):
    n = 0
    for m in unet.modules():
        if hasattr(m, "to_k") and isinstance(m.to_k, nn.Linear):
            m.to_k = LoRALinear(m.to_k, r=r); n += 1
        if hasattr(m, "to_v") and isinstance(m.to_v, nn.Linear):
            m.to_v = LoRALinear(m.to_v, r=r); n += 1
    return n


def ensure_lora_to_device(unet, device):
    for m in unet.modules():
        if isinstance(getattr(m, "to_k", None), LoRALinear):
            dev = m.to_k.base.weight.device; dty = m.to_k.base.weight.dtype
            m.to_k.A.to(device=dev, dtype=dty); m.to_k.B.to(device=dev, dtype=dty)
        if isinstance(getattr(m, "to_v", None), LoRALinear):
            dev = m.to_v.base.weight.device; dty = m.to_v.base.weight.dtype
            m.to_v.A.to(device=dev, dtype=dty); m.to_v.B.to(device=dev, dtype=dty)


class LastStateAdapter(nn.Module):
    def __init__(self, dim=768, hidden=1536):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),

            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),

            nn.Linear(hidden, dim, bias=True),
        )
        self.final_ln = nn.LayerNorm(dim) # mimics CLIP's final LayerNorm

    def forward(self, x):
        y = self.ff(x)
        y = self.final_ln(y)
        return y


class LastStateAdapter_ctrlora(nn.Module):
    def __init__(self, dim=768, hidden=768):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),

            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),

            nn.Linear(hidden, dim, bias=True),
        )
        self.final_ln = nn.LayerNorm(dim) # mimics CLIP's final LayerNorm

    def forward(self, x):
        y = self.ff(x)
        y = self.final_ln(y)
        return y

def lora_qkv(unet, r_q=4, r_kv=12):
    """给每个 cross-attn 模块挂 Q/K/V 的 LoRA。
    Q 的秩较小（r_q），K/V 秩较大（r_kv）。"""
    n = {"q": 0, "k": 0, "v": 0}
    for m in unet.modules():
        if hasattr(m, "to_q") and isinstance(m.to_q, nn.Linear):
            m.to_q = LoRALinear(m.to_q, r=r_q)
            n["q"] += 1
        if hasattr(m, "to_k") and isinstance(m.to_k, nn.Linear):
            m.to_k = LoRALinear(m.to_k, r=r_kv)
            n["k"] += 1
        if hasattr(m, "to_v") and isinstance(m.to_v, nn.Linear):
            m.to_v = LoRALinear(m.to_v, r=r_kv)
            n["v"] += 1
    return n


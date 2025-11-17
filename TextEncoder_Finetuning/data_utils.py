# data_utils.py
import json, os, random
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

random.seed(42)

def stratified_split(jsonl_path, out_dir, per_prompt_counts=(400, 50, 50)):
    """
    按 prompt 分层切分：每个 prompt 固定切成 train/val/test 数量
    per_prompt_counts: (train_n, val_n, test_n)
    生成 3 个 jsonl：pairs_train.jsonl / pairs_val.jsonl / pairs_test.jsonl
    """
    train_n, val_n, test_n = per_prompt_counts
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = [json.loads(l) for l in f if l.strip()]
    buckets = {}
    for j in lines:
        key = j.get("prompt", "")
        buckets.setdefault(key, []).append(j)

    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "train": os.path.join(out_dir, "pairs_train.jsonl"),
        "val":   os.path.join(out_dir, "pairs_val.jsonl"),
        "test":  os.path.join(out_dir, "pairs_test.jsonl"),
    }
    writers = {k: open(v, 'w', encoding='utf-8') for k, v in paths.items()}

    for prompt, items in buckets.items():
        random.shuffle(items)
        assert len(items) >= train_n + val_n + test_n, f"{prompt} amount not enough：{len(items)}"
        for j in items[:train_n]:
            writers["train"].write(json.dumps(j, ensure_ascii=False) + "\n")
        for j in items[train_n:train_n + val_n]:
            writers["val"].write(json.dumps(j, ensure_ascii=False) + "\n")
        for j in items[train_n + val_n:train_n + val_n + test_n]:
            writers["test"].write(json.dumps(j, ensure_ascii=False) + "\n")

    for w in writers.values():
        w.close()

    return paths


def stratified_split_ratio(jsonl_path, out_dir, per_prompt_ratio=(0.8, 0.1, 0.1),
                           min_per_split=(1, 1, 1), shuffle_seed=42, verbose=True):
    """
    按 prompt 分层，按比例切分 train/val/test。
    - per_prompt_ratio: (train, val, test) 之和应为 1.0（可以是 0.8, 0.1, 0.1）
    - min_per_split: 每个 prompt 各 split 的最小样本数（不足则置为 0 或按需要改）
    - 自动四舍五入；保证总数不超过该 prompt 的总样本数；余数给 train
    返回: {"train": path, "val": path, "test": path}
    """
    assert abs(sum(per_prompt_ratio) - 1.0) < 1e-6, "per_prompt_ratio 必须相加为 1.0"
    random.seed(shuffle_seed)

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = [json.loads(l) for l in f if l.strip()]
    buckets = {}
    for j in lines:
        key = j.get("prompt", "")
        buckets.setdefault(key, []).append(j)

    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "train": os.path.join(out_dir, "pairs_train.jsonl"),
        "val":   os.path.join(out_dir, "pairs_val.jsonl"),
        "test":  os.path.join(out_dir, "pairs_test.jsonl"),
    }
    writers = {k: open(v, 'w', encoding='utf-8') for k, v in paths.items()}

    tot_train = tot_val = tot_test = 0
    dropped = {}

    for prompt, items in buckets.items():
        random.shuffle(items)
        n = len(items)

        # 如果样本太少，直接全部给 train（或你可选择跳过该 prompt）
        if n < sum(min_per_split):
            for j in items:
                writers["train"].write(json.dumps(j, ensure_ascii=False) + "\n")
            tot_train += n
            dropped[prompt] = f"all->train (n={n})"
            continue

        # 先按比例取整
        t = int(round(n * per_prompt_ratio[0]))
        v = int(round(n * per_prompt_ratio[1]))
        s = n - t - v
        # 最小样本约束
        t = max(t, min_per_split[0])
        v = max(v, min_per_split[1])
        s = max(s, min_per_split[2])
        # 若超了总数，优先缩小 test/val
        while t + v + s > n and s > min_per_split[2]:
            s -= 1
        while t + v + s > n and v > min_per_split[1]:
            v -= 1
        while t + v + s > n and t > min_per_split[0]:
            t -= 1
        # 把剩余不足/富余分配给 train
        if t + v + s < n:
            t += (n - (t + v + s))

        assert t + v + s == n, f"split total mismatch for prompt={prompt}"

        # 写文件
        for j in items[:t]:
            writers["train"].write(json.dumps(j, ensure_ascii=False) + "\n")
        for j in items[t:t+v]:
            writers["val"].write(json.dumps(j, ensure_ascii=False) + "\n")
        for j in items[t+v:]:
            writers["test"].write(json.dumps(j, ensure_ascii=False) + "\n")

        tot_train += t; tot_val += v; tot_test += s

    for w in writers.values():
        w.close()

    if verbose:
        print(f"[split] train/val/test = {tot_train}/{tot_val}/{tot_test}")
        if dropped:
            print("[split] prompts with very few samples -> all to train:", dropped)

    return paths


class PairsJSONLDataset(Dataset):
    """
    读取 jsonl，其中 source/target 为相对 ROOTDIR 的路径：
      {"prompt": "...", "source": "source/xxx.png", "target": "target/xxx.png", ...}
    返回:
      - prompt: str
      - seg:    FloatTensor [C,H,W]  (0..1)
      - gt:     FloatTensor [3,H,W]  (-1..1)
    """
    def __init__(self, jsonl_path, root_dir, image_size=512, seg_mode="RGB"):
        self.items = []
        self.root_dir = root_dir
        self.seg_mode = seg_mode
        self.size = image_size

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                j = json.loads(line)
                prompt = j.get("prompt") or j.get("text") or ""
                src = j.get("source") or j.get("seg") or j.get("seg_path") or j.get("control")
                tgt = j.get("target") or j.get("gt")  or j.get("gt_path")
                assert src and tgt, f"missing source/target in line: {j}"
                src_path = src if os.path.isabs(src) else os.path.join(root_dir, src)
                tgt_path = tgt if os.path.isabs(tgt) else os.path.join(root_dir, tgt)
                self.items.append((prompt, src_path, tgt_path))

        I = transforms.InterpolationMode
        self.tf_seg = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=I.NEAREST),
            transforms.ToTensor(),     # 0..1
        ])
        self.tf_gt = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=I.BILINEAR),
            transforms.ToTensor(),     # 0..1
            transforms.Lambda(lambda x: x * 2 - 1),  # -> -1..1
        ])

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        prompt, pseg, pgt = self.items[idx]
        seg_img = Image.open(pseg).convert(self.seg_mode)
        gt_img  = Image.open(pgt).convert("RGB")
        seg = self.tf_seg(seg_img)
        gt  = self.tf_gt(gt_img)
        seg = seg.squeeze()  # 保证 [3, H, W]
        gt = gt.squeeze()
        return {"prompt": prompt, "seg": seg, "gt": gt, "seg_path": pseg, "gt_path": pgt}


class RepeatDataset(Dataset):
    def __init__(self, item_dict, repeat=1000):
        self.seg = item_dict["seg"]
        self.gt = item_dict["gt"]
        self.prompt = item_dict["prompt"]
        self.repeat = repeat
    def __len__(self): return self.repeat
    def __getitem__(self, idx):
        return {"seg": self.seg.clone(), "gt": self.gt.clone(), "prompt": self.prompt}

class RepeatPairsDataset(Dataset):
    """重复整个 dataset 若干次（用于多样本重复）"""
    def __init__(self, base_dataset, repeat=100):
        self.base = base_dataset
        self.repeat = repeat
        self.N = len(base_dataset)
    def __len__(self):
        return self.N * self.repeat
    def __getitem__(self, idx):
        return self.base[idx % self.N]
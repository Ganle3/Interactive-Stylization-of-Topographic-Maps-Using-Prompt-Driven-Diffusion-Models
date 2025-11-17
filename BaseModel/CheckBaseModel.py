import torch
ckpt_path = r"D:\Junyhuang\Project2\BaseModel\Swisstopo.ckpt"

state = torch.load(ckpt_path, map_location="cpu")  # 仅做键名检查，不占GPU
state = state.get("state_dict", state)
keys = list(state.keys())

has_control = any(k.startswith("control_") or k.startswith("control_model.") or "control" in k for k in keys)
print("Has ControlNet in ckpt? ->", has_control)

# 看看顶层前缀，心里有数
top_prefix = sorted(set(k.split('.')[0] for k in keys))
print(top_prefix[:20])
import os, math, json, sys
sys.path.append(r"D:\Junyhuang\Project2\ControlNet")

from ControlNet import config
from ControlNet.share import *
import einops
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import os
from pytorch_lightning import seed_everything
from ControlNet.annotator.util import resize_image, HWC3
from ControlNet.cldm.model import create_model, load_state_dict
from ControlNet.cldm.ddim_hacked import DDIMSampler
from PIL import Image
import time
start = time.time()

# Python script to evaluate any trained ControlNet model visually on the corresponding test set.
# This script was adopted and adapted from:
# https://github.com/lllyasviel/ControlNet/blob/main/gradio_seg2image.py

model = create_model('D:\Junyhuang\Project2\ControlNet\models\cldm_v15.yaml').cpu()
CKPT    = r"D:\Junyhuang\Project2\BaseModel\Swisstopo.ckpt"
model.load_state_dict(load_state_dict(CKPT, location='cuda'), strict=False)  # adjust path to model
model = model.cuda()
ddim_sampler = DDIMSampler(model)

# adjust prompt according to chosen model: 'map in swisstopo style' or 'map in old national style' or 'map in siegfried style'

# prompt = 'map in old national style'
# prompt = 'map in swiss topo style'
prompt = 'set forest to dark green color'
# a_prompt = 'best quality, extremely detailed'
n_prompt = ''
# prompt = ''
a_prompt = ''
# n_prompt = ''

image_resolution = 512
ddim_steps = 20
guess_mode = False
strength = 1
scale = 9
seed = 1286028432
eta = 0
num_samples = 1

imgShow = False  # set to True to view each generated tile together with the control image

# path_s = "OldNationalTestSet/source"  # adjust path to test set vector map tiles
path_s = r"D:\Junyhuang\Project2_Data\Training Data\Item_color\source"
dir_list_s = os.listdir(path_s)
source = [dir_list_s[0]]

source.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

resulting_tiles = []  # resulting tiles are later saved as numpy array and can be stitched together using stitchTiles.py
for s in source:

    input_img = Image.open(path_s + '/' + s)
    input_image = np.array(input_img)
    input_img.close()

    with torch.no_grad():

        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        control = torch.from_numpy(img.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control],
                "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control],
                   "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else (
                    [strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(
            np.uint8)

        results = [x_samples[i] for i in range(num_samples)]

    output = results[0]

    resulting_tiles.append(output)

    if(imgShow):

        f, axarr = plt.subplots(2, 1)

        axarr[0].imshow(input_image)
        axarr[1].imshow(output)
        plt.show()

res = np.array(resulting_tiles)
np.save("Tiles.npy", res)

# Save one of the generated results as an image for visual inspection
save_path = "myprompt_ControlNet_output_test1.png"

# If only one tile was processed, resulting_tiles[0] is the output image
Image.fromarray(resulting_tiles[0]).save(save_path)
print(f"Saved output image to {save_path}")


end = time.time()
print(end - start)
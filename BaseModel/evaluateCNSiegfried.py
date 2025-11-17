from share import *
import config
import einops
import numpy as np
import torch
import random
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from Unet import multi_unet_model
import os
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import keras

# Python script to evaluate the specialized Siegfried style model (or the combined model for Siegfried style) visually and quantitatively
# This script was adopted and adapted from:
# https://github.com/lllyasviel/ControlNet/blob/main/gradio_seg2image.py

model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./models/Siegfried.ckpt', location='cuda'),
                      strict=False)  # either Combined.ckpt model or Siegfried.ckpt model
model = model.cuda()
ddim_sampler = DDIMSampler(model)

n_classes = 9


def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=512, IMG_WIDTH=512, IMG_CHANNELS=3)


seg_model = get_model()  # get U-Net model for image segmentation
seg_model.load_weights('segmentationModelSiegfried.weights.h5')  # It is possible that this model only works using a very specific set up. Maybe it has to be retrained when using a different one.

prompt = 'map in siegfried style'
a_prompt = 'best quality, extremely detailed'
n_prompt = 'text, labels, annotations, numbers'

image_resolution = 512
ddim_steps = 20
guess_mode = False
strength = 1
scale = 9
seed = -1  # random seed
eta = 0
num_samples = 6  # set num_samples to > 1 to augment the final output with the help of the two implemented evaluation methods 'calculate_MIOU' and 'mask_check'

path_s = "sourceTest"  # vector map tiles
dir_list_s = os.listdir(path_s)
source = dir_list_s
source.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

path_mask = "siegfriedSegmGT"  # vector map tiles with colors mapped to the segmentation classes
dir_list_mask = os.listdir(path_mask)
mask = dir_list_mask
mask.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

m = keras.metrics.MeanIoU(num_classes=9, ignore_class=8)

resulting_tiles = []


def calculate_MIOU(segm, msk):
    scores = []

    msk[msk == 8] = 0
    masked_tile = msk.reshape(-1)

    for segmented_tile in segm:
        segmented_tile[segmented_tile == 8] = 0
        segmented_tile = segmented_tile.reshape(-1)

        m.update_state(masked_tile, segmented_tile)  # calculate MIoU score between segmented tile and GT mask
        scores.append(m.result().numpy())
        m.reset_state()

    return scores


def mask_check(prediction, control_img):
    pred_img = np.array(prediction)[:, :, :3]
    control_img = np.array(control_img)[:, :, :3]

    background_mask = cv2.inRange(control_img, np.array([255, 255, 255]), np.array([255, 255, 255])) / 255
    background_mask = cv2.merge((background_mask, background_mask, background_mask))
    a = background_mask.sum() / (512 * 512 * 3)

    building_mask = cv2.inRange(control_img, np.array([82, 82, 82]), np.array([82, 82, 82])) / 255
    building_mask = cv2.merge((building_mask, building_mask, building_mask))
    b = building_mask.sum() / (512 * 512 * 3)

    forest_mask = cv2.inRange(control_img, np.array([77, 175, 74]), np.array([77, 175, 74])) / 255
    forest_mask = cv2.merge((forest_mask, forest_mask, forest_mask))
    c = forest_mask.sum() / (512 * 512 * 3)

    control_bg = np.where(control_img == [255, 255, 255], [240, 238, 223], np.nan)
    img_background = (pred_img * background_mask)
    score_a = np.nanmean((control_bg - img_background) ** 2)

    control_build = np.where(control_img == [82, 82, 82], [16, 17, 13], np.nan)
    img_buildings = (pred_img * building_mask)
    score_b = np.nanmean((control_build - img_buildings) ** 2)

    control_forest = np.where(control_img == [77, 175, 74], [240, 238, 223], np.nan)
    img_forest = (pred_img * forest_mask)
    score_c = np.nanmean((control_forest - img_forest) ** 2)

    # Edge cases

    if (np.isnan(a) or np.isnan(score_a)):
        a = 0
        score_a = 0

    if (np.isnan(b) or np.isnan(score_b)):
        b = 0
        score_b = 0

    if (np.isnan(c) or np.isnan(score_c)):
        c = 0
        score_c = 0

    print(a, b, c)
    print(a * score_a + b * score_b + c * score_c)  # calculate overall MSE using a weighted sum
    print("----------------")

    return a * score_a + b * score_b + c * score_c


max_miou_scores = []
min_mse_scores = []

for s in source:

    input_img = Image.open(path_s + '/' + s)
    control_image = np.array(input_img)
    input_img.close()

    mask_img = Image.open(path_mask + '/' + s)
    control_mask = np.array(mask_img)
    mask_img.close()

    with torch.no_grad():

        img = resize_image(HWC3(control_image), image_resolution)
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
                "c_crossattn": [model.get_learned_conditioning([prompt  + ', ' + a_prompt] * num_samples)]}
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
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0,
                                                                                                           255).astype(
            np.uint8)

        results = [x_samples[i] for i in range(num_samples)]

    predictions = []
    layout_scores = []

    for pred in results:
        score = mask_check(pred, control_image)
        layout_scores.append(score)
        predictions.append(pred / 255.)

    min_labels = np.argmin(layout_scores)
    min_mse_scores.append(np.min(layout_scores))
    print(np.min(layout_scores))
    print('Best Tile: ', min_labels)

    # f, axarr = plt.subplots(num_samples * 2, 1)

    # for i in range(num_samples):
    # axarr[i].imshow(results[i])

    segmentations = []

    for tile in predictions:
        tile = tile.reshape(-1, 512, 512, 3)
        y_pred = seg_model.predict(tile)
        seg_img = np.argmax(y_pred, axis=3)[0, :, :]
        segmentations.append(seg_img)

    # for i in range(num_samples, num_samples*2):
    # axarr[i].imshow(segmentations[i - num_samples], cmap='jet')

    miou_scores = calculate_MIOU(segmentations, control_mask)
    print(miou_scores)

    # plt.show()
    max_miou_scores.append(np.max(miou_scores))
    print(np.max(miou_scores))
    max_miou = np.argmax(miou_scores)

    if max_miou == min_labels:  # if the two methods agree -> choose this tile

        best_tile = results[max_miou]

    else:  # methods don't agree -> choose best tile according to check_mask method

        best_tile = results[min_labels]

    # plt.imshow(best_tile)
    # plt.show()

    resulting_tiles.append(best_tile)

print("Average MSE: ", np.mean(min_mse_scores))
print("Average MIoU score: ", np.mean(max_miou_scores))
res = np.array(resulting_tiles)
np.save("TilesSiegfried2.npy", res)  # all augmented tiles saved


import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import json
import matplotlib.image as mpimg

#  Python script to create the needed triple structure (source, target, prompt.json) to train ControlNet



# Lists with file-paths to the large vector (source_list) and raster (target_list) images.
# For Swisstopo style use contents of "\Data\LargeSheets\SwisstopoLargeSheets"
# For Old National style use contents of "\Data\LargeSheets\OldNationalLargeSheets"
# For Siegfried style use contents of "\Data\LargeSheets\SiegfriedLargeSheets"

# Example for Swisstopo:
'''
sources_list = ["sourceSwiss.png", "sourceSwiss1.png", ...]
targets_list = ["targetSwiss.png", "targetSwiss1.png", ...]

'''

sources_list = ["test_oldNational.png"]

output_folder_source = "TileOldnationaltest"  # temporary folders since additional processing steps are needed
os.makedirs(output_folder_source, exist_ok=True)

def process(S):
    h = S.shape[0] % dim
    w = S.shape[1] % dim

    height = S.shape[0] - h
    width = S.shape[1] - w

    S = S[:height, :width, :]  # make shape divisible by 512


    n_h = int(S.shape[0] / 512)
    n_w = int(S.shape[1] / 512)

    print("Tiling started ")

    source_slices = np.split(S, n_h)

    source_slices_list = []


    for i in tqdm(
            range(n_h)
    ):  # split horizontal slices vertically to create n_h x n_w tiles of size 512 x 512

        source_slices_list.append(np.array(np.split(source_slices[i], n_w, axis=1)))

    print("Tiling done: ", n_h * n_w, "tiles created")
    source = np.array(source_slices_list)

    source = source.reshape((n_h * n_w, dim, dim, 3))
    n_tiles = n_h * n_w


    for j in tqdm(range(0, n_tiles)):

        source_img = source[j]


        # plt.imshow(source_img)
        # plt.show()


        source_filename = f"{j}.png"

        output_path = os.path.join(output_folder_source, source_filename)
        plt.imsave(output_path, source_img)


dim = 512  # required shape: 512 x 512 pixels

for item in range(len(sources_list)):



    s = cv2.imread(sources_list[item])

    s = cv2.cvtColor(s, cv2.COLOR_BGR2RGB)

    s = np.array(s)


    sour = s

    process(sour)



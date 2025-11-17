import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from tqdm import tqdm
from PIL import Image

# Python script to create ground-truth masks for training the U-Net or evaluating ControlNet for Siegfried style.
# The RGB values of the input vector map tiles (located in the source folders) are mapped to unique classes

siegfriedSource = False  # Set to True to create GT masks for training the U-Net

if siegfriedSource:

    # create GT mask to train U-Net

    path = "source"
    end_path = "source"
    dir_list_t = os.listdir(path)
    target = dir_list_t
    target.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    index = 0
    count = 0

    for img in tqdm(target):
        index = img

        image = Image.open(path + '/' + img)
        img = np.array(image)[:,:,:3]
        image.close()

        lower_bound = np.array([5, 5, 5])
        upper_bound = np.array([5, 5, 5])
        mask_building = cv2.inRange(img, lower_bound, upper_bound)

        lower_bound = np.array([255, 255, 255])
        upper_bound = np.array([255, 255, 255])
        mask_background = cv2.inRange(img, lower_bound, upper_bound)

        lower_bound = np.array([77, 175, 74])
        upper_bound = np.array([77, 175, 74])
        mask_forest = cv2.inRange(img, lower_bound, upper_bound)

        lower_bound = np.array([149, 74, 162])
        upper_bound = np.array([149, 74, 162])
        mask_roads = cv2.inRange(img, lower_bound, upper_bound)

        lower_bound = np.array([63, 96, 132])
        upper_bound = np.array([63, 96, 132])
        mask_stream = cv2.inRange(img, lower_bound, upper_bound)

        lower_bound = np.array([55, 126, 184])
        upper_bound = np.array([55, 126, 184])
        mask_lake = cv2.inRange(img, lower_bound, upper_bound)

        lower_bound = np.array([96, 147, 201])
        upper_bound = np.array([96, 147, 201])
        mask_river = cv2.inRange(img, lower_bound, upper_bound)

        lower_bound = np.array([255, 0, 0])
        upper_bound = np.array([255, 0, 0])
        mask_paths = cv2.inRange(img, lower_bound, upper_bound)

        lower_bound = np.array([247, 128, 30])
        upper_bound = np.array([247, 128, 30])
        mask_ignore = cv2.inRange(img, lower_bound, upper_bound)
        
        # assign unique class values

        mask = np.zeros((512, 512))
        mask[mask_building == 255] = 5
        mask[mask_background == 255] = 4
        mask[mask_forest == 255] = 1
        mask[mask_roads == 255] = 3
        mask[mask_stream == 255] = 2
        mask[mask_river == 255] = 6
        mask[mask_paths == 255] = 7
        mask[mask_ignore == 255] = 8
        mask[mask == 0] = 4
        mask[mask_lake == 255] = 0

        #plt.imshow(mask)
        #plt.show()

        cv2.imwrite(end_path + '/' + index, mask)

else:

    # to create GT mask for evaluation of ControlNet for Siegfried style with U-Net

    path = "source"
    end_path = "source"
    dir_list_t = os.listdir(path)
    target = dir_list_t
    target.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    index = 0
    count = 0

    for img in tqdm(target):
        index = img

        image = Image.open(path + '/' + img)
        img = np.array(image)[:, :, :3]
        image.close()

        #plt.imshow(img)
        # plt.show()

        lower_bound = np.array([82, 82, 82])
        upper_bound = np.array([82, 82, 82])
        mask_building = cv2.inRange(img, lower_bound, upper_bound)

        lower_bound = np.array([255, 255, 255])
        upper_bound = np.array([255, 255, 255])
        mask_background = cv2.inRange(img, lower_bound, upper_bound)

        lower_bound = np.array([77, 175, 74])
        upper_bound = np.array([77, 175, 74])
        mask_forest = cv2.inRange(img, lower_bound, upper_bound)

        lower_bound = np.array([149, 74, 162])
        upper_bound = np.array([149, 74, 162])
        mask_roads = cv2.inRange(img, lower_bound, upper_bound)

        lower_bound = np.array([63, 96, 132])
        upper_bound = np.array([63, 96, 132])
        mask_stream = cv2.inRange(img, lower_bound, upper_bound)

        lower_bound = np.array([55, 126, 184])
        upper_bound = np.array([55, 126, 184])
        mask_lake = cv2.inRange(img, lower_bound, upper_bound)

        lower_bound = np.array([96, 147, 201])
        upper_bound = np.array([96, 147, 201])
        mask_river = cv2.inRange(img, lower_bound, upper_bound)

        lower_bound = np.array([0, 0, 0])
        upper_bound = np.array([0, 0, 0])
        mask_paths = cv2.inRange(img, lower_bound, upper_bound)

        lower_bound = np.array([255, 0, 0])
        upper_bound = np.array([255, 0, 0])
        mask_ignore = cv2.inRange(img, lower_bound, upper_bound)
        
        # assign unique class values

        mask = np.zeros((512, 512))
        mask[mask_building == 255] = 5
        mask[mask_background == 255] = 4
        mask[mask_forest == 255] = 1
        mask[mask_roads == 255] = 3
        mask[mask_stream == 255] = 2
        mask[mask_river == 255] = 6
        mask[mask_paths == 255] = 7
        mask[mask_ignore == 255] = 8
        mask[mask == 0] = 4
        mask[mask_lake == 255] = 0

        #plt.imshow(mask)
        #plt.show()

        cv2.imwrite(end_path + '/' + index, mask)
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2

#  Python script to add the contour lines to the 5120 x 5120 pixel Siegfried Map, completing the style.

img = Image.open("Siegfried_Results_Raw.png")  # generated Siegfried Map
heights = Image.open("siegfriedContourLines.png")  # corresponding contour lines (image showing the vector data)

image = np.array(img)
hts = np.array(heights)
img.close()
heights.close()

brown = np.array([184, 94, 20, 255])  # average color of the original Siegfried Map contour lines
black_low = np.array([0, 0, 0, 255])  # color range of black pixels to later not place contour lines on top of buildings
black_up = np.array([50, 50, 50, 255])

buildings_mask = cv2.inRange(image, black_low, black_up)
buildings_mask = np.where(buildings_mask == 255, True, False)

hts[buildings_mask] = 0
heights_mask = np.all(hts == brown, axis=-1)

image[heights_mask] = hts[heights_mask]

plt.imshow(image)
plt.show()

plt.imsave('Siegfried_Results_Raw_Contours.png', image)
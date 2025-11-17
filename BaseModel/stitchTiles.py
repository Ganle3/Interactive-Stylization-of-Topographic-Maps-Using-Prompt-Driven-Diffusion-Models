import numpy as np
import matplotlib.pyplot as plt

#  Python script to stitch together 100 generated map tiles. Used to create the large 5120 x 5120 pixel output maps.

a = np.load("siegfriedCombinedResults.npy")  # insert NPY array containing 100 tiles
print(a.shape)

img_hz = []

offset = 0

for i in range(10):

    first = a[offset]

    for j in range(10):

        if j == 1:

            img = np.hstack((first, a[j + offset]))

        elif j > 1:

            img = np.hstack((img, a[j + offset]))

    img_hz.append(img)
    offset += 10


first_row = img_hz[0]

for index, hz_tiles in enumerate(img_hz):

    if index == 1:

        img_full = np.vstack((first_row, hz_tiles))

    elif index > 1:

        img_full = np.vstack((img_full, hz_tiles))


plt.imshow(img_full)
plt.show()
plt.imsave('siegfriedCombined.png', img_full)
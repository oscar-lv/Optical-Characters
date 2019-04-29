# Sample code for AI Coursework
# Extracts the letters from a printed page.

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
# Keep in mind that after you extract each letter, you have to normalise the size.
# You can do that by using scipy.imresize. It is a good idea to train your classifiers
# using a constast size (for example 20x20 pixels)
import numpy as np
from scipy.misc import imread
from skimage.measure import regionprops
from skimage.morphology import label
from skimage.segmentation import clear_border

image = imread('./o.png', 1)

# apply threshold in order to make the image binary
bw = image < 230

# remove artifacts connected to image border
cleared = bw.copy()
clear_border(cleared)

# label image regions
label_image = label(cleared, neighbors=8)
borders = np.logical_xor(bw, cleared)
label_image[borders] = -1

print(label_image.max())

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
ax.imshow(bw, cmap='jet')

l = []
for region in regionprops(label_image):
    # skip small images
    if region.area > 2500:
        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        l.append(region)

ax.imshow(l[0].image, cmap='jet')

plt.show()

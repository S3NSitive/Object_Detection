import scipy
import imageio
import numpy as np
import scipy.signal as sig
import skimage.segmentation

from matplotlib import pyplot as plt

"""
scipy.misc.imread and imresize have been removed. imread uses the Python Imaging Library (PIL) to read an image.
We can use plt.imread or imageio.imread
"""
# img2 = scipy.misc.imread("data/manu-2013.jpg", mode="L")
img2 = plt.imread("data/manu-2013.jpg")
# img2 = imageio.imread("data/manu-2013.jpg")

segment_mask1 = skimage.segmentation.felzenszwalb(img2, scale=100)
segment_mask2 = skimage.segmentation.felzenszwalb(img2, scale=1000)

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.imshow(segment_mask1); ax1.set_xlabel("k=100")
ax2.imshow(segment_mask2); ax2.set_xlabel("k=1000")
fig.suptitle("Felsenszwalb's efficient graph based image segmentation")
plt.tight_layout()
plt.show()

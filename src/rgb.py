#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# In[ ]:


img = Image.open('blue_green.jpg')
img_arr = np.array(img)

def rgb2gray(rgb):
    # :3 to ignore alpha channel (if exists)
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

naive_gray = np.dot(img_arr[...,:3], np.array([1, 1, 1]) / 3).astype(np.uint8) # np.dot collapses a dimension
# grayscale = 0.299*img_arr[...,0] + 0.587*img_arr[..., 1] + 0.114*img_arr[...,2]
grayscale = rgb2gray(img_arr)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].imshow(img_arr)
ax[0].set_title("Original")

ax[1].imshow(naive_gray, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
ax[1].set_title("Naive Gray")

ax[2].imshow(grayscale, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
ax[2].set_title("Lum. Gray")


# In[ ]:





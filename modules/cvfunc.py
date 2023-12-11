"""
This module is used to define all functions which perform opencv functions
"""


import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import gaussian_filter

"""
This function takes the output mask from the Unet seg model as input
reduces the width to 1 px for further processing. 
"""


def reduce_width(output_mask):
    # making sure the input image is in the right format for being thresholded
    if output_mask.dtype != np.uint8:
        output_mask = np.clip(output_mask, 0, 255).astype(np.uint8)
        
    if len(output_mask.shape) == 3 and output_mask.shape[2] == 3:
        output_mask = cv2.cvtColor(output_mask, cv2.COLOR_BGR2GRAY)
    
    # Apply a Gaussian blur to smooth out the image before skeletonization
    blurred_image = gaussian_filter(output_mask, sigma=1)
    
    # Apply threshold to ensure the image is binary
    _, binary_image = cv2.threshold(blurred_image, 127, 255, cv2.THRESH_BINARY)
    
    # Convert to boolean array for skeletonization
    binary_image = binary_image > 0
    
    # Perform skeletonization which reduces the width of lines to 1px
    skeleton = skeletonize(binary_image)
    
    # Convert back to format for further processing
    skeletonized_image  = (skeleton * 255).astype(np.uint8)
    
    return skeletonized_image


import cv2
import tifffile
import numpy as np


image_path = "batman.jpg"
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # load in image

# resize image
size = (64, 64)
image_tiny = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)

image_tiny = cv2.cvtColor(image_tiny, cv2.COLOR_BGR2GRAY)               # convert to grayscale
image_tiny = (image_tiny / 255.0).astype(np.float32)                    # normalize, assume 8-bit image

image_binary = (image_tiny > 0.5)                                       # binarize

image_viewable = (image_binary.astype(np.uint8) * 255)                  # make image viewable as uint8 [0, 255]

cv2.imwrite('./tiny_image.png', image_viewable)                         # save [0-255] binary image png
tifffile.imwrite('./tiny_image.tiff', image_binary)                     # save [0,1] binary image tiff
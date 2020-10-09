import numpy as np
from PIL import Image as IMG
import cv2
from skimage.io import imread, imshow
from scipy.stats import itemfreq

def dominant_color(img):
    # img should be the img object or path to the image
    # read in image using openCV
    img = cv2.imread(img)
    # convert to float32
    img = np.float32(img)
    # change to shape for k-means clustering
    pixels = img.reshape((-1, 3))
    # choose clusters
    n_colors = 5
    # set critera for clustering using openCV, look at their kmeans documentation
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, clusters = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    # change to 8-bit unsigned integer value for RGB values
    palette = np.uint8(clusters)
    res = palette[labels.flatten()]
    res = res.reshape(img.shape)
    # grab top one
    RGB_Values = palette[np.argmax(itemfreq(labels)[:, -1])]

    return RGB_Values

# To test this function, set up your img object or path
# img = '/path/img'

# gdc=get_dominant_color(img)
# print(gdc)

# dom_red = np.round(gdc[0]/255,2)
# dom_green = np.round(gdc[1]/255,2)
# dom_blue = np.round(gdc[2]/255,2)

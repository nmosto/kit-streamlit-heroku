import numpy as np
from PIL import Image as IMG
from skimage import feature

def uniformity(img):
    #path = images_path + img
    im = IMG.open(img)
    im_array = np.asarray(im.convert(mode='L'))
    edges_sigma1 = feature.canny(im_array, sigma=3)
    apw = (float(np.sum(edges_sigma1)) / (im.size[0]*im.size[1]))
    return apw*100

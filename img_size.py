import os
import numpy as np
from PIL import Image as IMG

def compression_size(img):
    #filename = images_path + filename
    size = os.stat(img)
    return size.st_size

def get_dimensions(img):
    #filename = images_path + filename
    img_size = IMG.open(img).size
    return img_size

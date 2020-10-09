import cv2

def average_color(img):
    #path = images_path + img
    img = cv2.imread(img)
    average_color = [img[:, :, i].mean() for i in range(img.shape[-1])]
    return average_color

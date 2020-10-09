import cv2

def blurriness(img):
    # img should be the img object or path to the image
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Take the variance of the laplacian to get a sense of the blurriness
    Laplacian = cv2.Laplacian(img, cv2.CV_64F).var()
    return Laplacian

# To test this function, set up your img object or path
# img = '/path/img'

# gbs=get_blurriness_color(img)
# print(gbs)



# Reference: https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/

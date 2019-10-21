import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import math
import numpy as np


import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
from skimage.viewer import ImageViewer
import os

# test to make sure plots are showing
# remember: "ssh -Y -i ..." to ssh in then do export DISPLAY=localhost:10.0
# plt.interactive(False)
#
# plt.hist(np.random.randn(100))
# plt.show()

path_dir_train = '/home/ubuntu/Deep-Learning/plant-classification/load_data/train/'

image_path = path_dir_train + 'Asclepias tuberosa/Asclepias curassavica.94.jpg'

img = cv2.imread(image_path)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)


#thresholding
#here 0 means that the image is loaded in gray scale format
# gray_image = cv2.imread('index.png',0)

ret,thresh_binary = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)
ret,thresh_binary_inv = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY_INV)
ret,thresh_trunc = cv2.threshold(img_gray,127,255,cv2.THRESH_TRUNC)
ret,thresh_tozero = cv2.threshold(img_gray,127,255,cv2.THRESH_TOZERO)
ret,thresh_tozero_inv = cv2.threshold(img_gray,127,255,cv2.THRESH_TOZERO_INV)



# use the following to visualize the image augmentations
names_list = ['oiriginal','gray','hsv','binary_thresh','binary_inv_thresh','trunc_thresh','tozero_thresh','tozero_inv_thresh']
images_list = [img,img_gray,img_hsv,thresh_binary,thresh_binary_inv,thresh_trunc,thresh_tozero,thresh_tozero_inv]

def visualize_augmentations(names,images):
    # fig, ax = plt.subplots(len(names)/3, 3, sharey=True, figsize=(15,15))
    for i in range(len(names)):
        plt.subplot(math.ceil(len(names)/3),3,i+1),plt.imshow(images[i],'gray')
        plt.title(names[i])
        plt.xticks([]),plt.yticks([])
        plt.show()

# visualize_augmentations(names_list,images_list)

# cv2.imwrite('test_write.jpg',image)



# edge detection
edges = cv2.Canny(img,100,200)
# plot edges
plt.imshow(edges)
plt.show()
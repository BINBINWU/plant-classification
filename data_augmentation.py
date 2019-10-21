import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# plt.interactive(False)
import cv2
import math
import numpy as np
import os

# test to make sure plots are showing
# remember: "ssh -Y -i ..." to ssh in then do export DISPLAY=localhost:10.0

#
# plt.hist(np.random.randn(100))
# plt.show()

def data_augmentation(dir_path):
    img_files = os.listdir(dir_path)

    for file in img_files:
        image_path = dir_path + file
        # read image
        img = cv2.imread(image_path)
        # convert color bgr to rgb (test image read in with default bgr makes yellow flowers look blue)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # convert image to gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # convert image to hsv
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # edge detection on image
        edges = cv2.Canny(img, 100, 200)

        # thresholding
        ret, thresh_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
        ret, thresh_binary_inv = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
        ret, thresh_trunc = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRUNC)
        ret, thresh_tozero = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO)
        ret, thresh_tozero_inv = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO_INV)

        img_output_list = [img_gray, img_hsv, edges, thresh_binary, thresh_binary_inv, thresh_trunc, thresh_tozero,
                       thresh_tozero_inv]
        outnames_list = ['gray', 'hsv', 'edges', 'binary_thresh', 'binary_inv_thresh', 'trunc_thresh',
                      'tozero_thresh', 'tozero_inv_thresh']
        for i in range(len(outnames_list)):
            outname = image_path.replace('.jpg',outnames_list[i] + '.jpg' )
            cv2.imwrite(outname,image_output_list[i])

path_dir_train = '/home/ubuntu/Deep-Learning/plant-classification/load_data/train/'




# image_path = path_dir_train + 'Asclepias tuberosa/Asclepias curassavica.94.jpg'
#read image
# img = cv2.imread(image_path)
# #convert color bgr to rgb (test image read in with default bgr makes yellow flowers look blue)
# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# # convert image to gray
# img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# #convert image to hsv
# img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# #edge detection on image
# edges = cv2.Canny(img,100,200)
#
# #thresholding
# ret,thresh_binary = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)
# ret,thresh_binary_inv = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY_INV)
# ret,thresh_trunc = cv2.threshold(img_gray,127,255,cv2.THRESH_TRUNC)
# ret,thresh_tozero = cv2.threshold(img_gray,127,255,cv2.THRESH_TOZERO)
# ret,thresh_tozero_inv = cv2.threshold(img_gray,127,255,cv2.THRESH_TOZERO_INV)
#
#
# # use the following to visualize the image augmentations
# names_list = ['oiriginal','gray','hsv','edges','binary_thresh','binary_inv_thresh','trunc_thresh','tozero_thresh','tozero_inv_thresh']
# images_list = [img,img_gray,img_hsv,edges,thresh_binary,thresh_binary_inv,thresh_trunc,thresh_tozero,thresh_tozero_inv]
#
# def visualize_augmentations(names,images):
#
#     for i in range(len(names)):
#         plt.subplot(math.ceil(len(names)/3),3,i+1),plt.imshow(images[i],'gray')
#         plt.title(names[i])
#         plt.xticks([]),plt.yticks([])
#         plt.show()
#
# # visualize_augmentations(names_list,images_list)

# cv2.imwrite('test_write.jpg',image)
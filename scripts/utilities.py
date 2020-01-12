#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import cv2

def save_image_RGB_or_gray(img, dir_out, file_name_out):
    """
    """
    # Save images using cv2.imwrite(), which require BGR color layout rather than RGB
    if len(img.shape) > 2 and img.shape[2] == 3:
        img_BGR = cv2.cvtColor( img , cv2.COLOR_RGB2BGR)
    else:
        img_BGR = img
    cv2.imwrite(dir_out + file_name_out, img_BGR)

def save_image_BGR_or_gray(img, dir_out, file_name_out):
    """
    """
    # Save images using cv2.imwrite(), which require BGR color layout rather than RGB
    cv2.imwrite(dir_out + file_name_out, img)

def functional_test_of_image(dir_in, file_name_in, dir_out, function_to_test):
    """
    The function_to_test is the following function
        img_dict = function_to_test(img)
    where img is the input image, and img_dict is an output dictionary of image
    """
    # Process one image
    #-----------#
    # Read an image
    print("Processing image file %s" % file_name_in)
    img_ori = mpimg.imread(dir_in + file_name_in) # Read image from disk

    # The function to test
    img_dict = function_to_test(img_ori)


    # Show all the intermediate images in pipeline
    #-------------------------------------------------#
    img_dict_key= list(img_dict.keys())
    print("img_dict_key = %s" % str(img_dict_key))
    fig_id = 0
    for fig_id in range(len(img_dict_key)):
        plt.figure(fig_id)
        plt.title("Fig. %d %s" % (fig_id, str(img_dict_key[fig_id])))
        plt.imshow(img_dict[img_dict_key[fig_id]], cmap='gray')
        # Save images using cv2.imwrite(), which require BGR color layout rather than RGB
        save_image_RGB_or_gray(img_dict[img_dict_key[fig_id]], dir_out, f_name[:-4] + "_" + img_dict_key[fig_id] + ".jpg" )

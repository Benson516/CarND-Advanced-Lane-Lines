#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import cv2
import glob
#
from utilities import *
import CAMERA


class IMAGE_WARPER(object):
    """
    """
    def __init__(self):
        """
        """
        # Parameters
        #-------------------------#
        img_size = (1280, 720) # (width, height)
        # Define 4 source/destination points np.float32([[,],[,],[,],[,]])
        ori_x1 = 200
        ori_x2 = img_size[0] - ori_x1
        self.warp_src = np.float32([ [593,450],[686,450],[ori_x2, img_size[1] ],[ori_x1,img_size[1] ] ])
        tranx_x1 = 350
        trans_x2 = img_size[0] - tranx_x1
        self.warp_dst = np.float32([[tranx_x1,0],[trans_x2,0],[trans_x2, img_size[1]],[tranx_x1, img_size[1]]])

        #
        # Define conversions in x and y from pixels space to meters
        self.ym_per_pix = 30/720 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/700 # meters per pixel in x dimension
        #-------------------------#

        # Variables
        #-------------------------#
        self.M_birdeye = None
        self.M_inv_birdeye = None
        #-------------------------#

        # Preparation
        self.cal_transform_matrices()

    def cal_transform_matrices(self):
        """
        """
        # Calculate the transform matrix
        self.M_birdeye = cv2.getPerspectiveTransform(self.warp_src, self.warp_dst)
        self.M_inv_birdeye = cv2.getPerspectiveTransform(self.warp_dst, self.warp_src)


    def transform(self, img):
        """
        """
        img_size = (img.shape[1], img.shape[0])
        return cv2.warpPerspective(img, self.M_birdeye, img_size, flags=cv2.INTER_LINEAR)

    def inverse_transform(self, img):
        """
        """
        img_size = (img.shape[1], img.shape[0])
        return cv2.warpPerspective(img, self.M_inv_birdeye, img_size, flags=cv2.INTER_LINEAR)





if __name__ == "__main__":
    # Create the camera object instance
    cam_1 = CAMERA.CAMERA()

    # Calibrate the camera
    cal_images = glob.glob("../camera_cal/calibration*.jpg")
    result = cam_1.calibrate(cal_images, (9,6))
    print("Calibration finished!" if result else "Calibration failed!")


    # Create an instance
    birdeye_trans = IMAGE_WARPER()


    # Read an image
    dir_in = "../test_images/"
    dir_out = "../output_images/warp/"
    files = sorted(os.listdir(dir_in))

    # Read an image
    f_name = files[0] # 0 and 1
    img_ori = mpimg.imread(dir_in + f_name) # Read image from disk
    img_undistorted = cam_1.undistort(img_ori)


    img_line = np.copy(img_undistorted)
    cv2.polylines(img_line, np.int32([birdeye_trans.warp_src]), isClosed=True, color=(255, 0, 0), thickness=1)
    img_birdeye = birdeye_trans.transform(img_line)

    # Save results
    save_image_RGB_or_gray(img_line, dir_out, f_name[:-4] + "img_line.jpg")
    save_image_RGB_or_gray(img_birdeye, dir_out, f_name[:-4] + "img_birdeye.jpg")



    fig_id = 0
    # Ploting
    plt.figure(fig_id); fig_id += 1
    plt.imshow(img_line, cmap='gray')
    plt.grid()
    # Save the resulted figure
    plt.savefig(dir_out + "plot_" +  f_name[:-4] + "_line.png")

    # Ploting
    plt.figure(fig_id); fig_id += 1
    plt.imshow(img_birdeye, cmap='gray')
    plt.grid()
    # Save the resulted figure
    plt.savefig(dir_out + "plot_" + f_name[:-4] + "_birdeye.png")


    plt.show()

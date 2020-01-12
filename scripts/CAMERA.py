#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import cv2
import glob

class CAMERA(object):
    """
    """
    def __init__(self):
        """
        """
        self.mtx = None
        self.dist = None

    def calibrate(self, image_name_list, board_size=(8,6)):
        """
        Inputs:
            - image_name_list: a list of image names
            - board_size: a touple of board size, like (8,6)
        Outputs:
            - True/False: True - finished, False - failed
        """
        objpoints = [] # 3D points in real-world coordinate
        imgpoints = [] # 2D points on 2D image plane

        # Prepare object points (of chessboard) in one image, like: [0,0,0], [1,0,0], [2,0,0], ..., [7,5,0]
        objp = np.zeros((board_size[0]*board_size[1], 3), np.float32)
        objp[:,:2] = np.mgrid[:board_size[0], :board_size[1]].T.reshape(-1,2) # Generate x,y coordinate and fill in the objp

        # Loop over images
        #--------------------------------------#
        gray = None
        for fname in image_name_list:
            # Read an image
            img = cv2.imread(fname)

            # Convert the current image to grayscale
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # Find chessboar corners
            ret, corners = cv2.findChessboardCorners(gray, board_size, None)

            # If objects are found, add object points and image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
        #--------------------------------------#

        # Calibrate camera
        #--------------------------------------#
        if gray is None:
            return False
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        self.mtx = mtx
        self.dist = dist
        #--------------------------------------#
        return True

    def undistort(self, img):
        """
        Input:
            - Original image
        Output:
            - Un-distorted image
        """
        mtx = None
        return cv2.undistort(img, self.mtx, self.dist, None, mtx)


if __name__ == "__main__":
    # Create the camera object instance
    cam_1 = CAMERA()

    # Calibrate the camera
    cal_images = glob.glob("../camera_cal/calibration*.jpg")
    # print("type(cal_images) = %s" % str(type(cal_images)))
    # print("len(cal_images) = %d" % len(cal_images))
    # print("cal_images = %s" % str(cal_images))
    result = cam_1.calibrate(cal_images, (9,6))
    print("Calibration finished!" if result else "Calibration failed!")

    # Test the undistort() function
    #-----------#
    dir_in = "../camera_cal/"
    dir_out = "../output_images/calibration/"
    f_name = "calibration1.jpg"
    print("Processing image file %s" % f_name)
    img_ori = mpimg.imread(dir_in + f_name) # Read image from disk
    img_undistorted = cam_1.undistort(img_ori)

    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img_ori)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(img_undistorted)
    ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    # Save the result figure
    plt.savefig(dir_out + "undistort_result.png")

    #
    plt.show()

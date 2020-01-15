# Self-Driving Car Engineer Nanodegree


## Project 2: **Advanced Lane Finding** 

The goals / steps of this project are the following:
- Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
- Apply a distortion correction to raw images.
- Use color transforms, gradients, etc., to create a thresholded binary image.
- Apply a perspective transform to rectify binary image ("birds-eye view").
- Detect lane pixels and fit to find the lane boundary.
- Determine the curvature of the lane and vehicle position with respect to center.
- Warp the detected lane boundaries back onto the original image.
- Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
---

The above steps can can be catogorized into two parts with the following structure

1. Off-line preparation
    - Camera calibration
2. On-line processing/pipeline
    - Image preprocessing
        - Un-distort an image
        - Create a binary image marking possible lane pixels
        - Apply perspective transform to "birds-eye view"
    - Lane-finding algorithm
        - Detect lane pixels and fit to find the lane boundary
        - Determine the curvature of the lane and vehicle position with respect to center
    - Visualization
        - Warp the detected lane boundaries back onto the original image
        - Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position


All codes of this project were written in iPython notebook named `project2_advanced_lane_finding.ipynb` located at the root folder of this project. The notebook has the structure mentioned above.

---

[//]: # (Image References)

[image1]: ./output_images/calibration/undistort_result_1.png "Undistorted"
[image2]: ./output_images/image_preprocessing/test4_preProc_img_undistorted.jpg "Road Transformed"
[image3-1]: ./output_images/lane_line_mask/test4_biProc_bi_yellow.jpg "Yellow binary Example"
[image3-2]: ./output_images/lane_line_mask/test4_biProc_bi_white.jpg "White binary Example"
[image3]: ./output_images/lane_line_mask/test4_biProc_img_out.jpg "Combined binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

This file is the writeup for this project.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The camera calibration step meants to find the intrinsic parameters and distortion coefficients of the camera we used. Because each camera is unique, and these parameters should be globally avelable for each camera, I create a class for cameras. The code of this step is writen in the notebook as a Python class called `CAMERA` in `Part 1: Off-line Preparation / Camera Calibration` section.
  
In the calibrate() method, I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

To demonstrate the usage of the `CAMERA` class, the code is simplyfy as follow. Full code can be checked in notebook.

```python
class CAMERA(object):
    ...
    def calibrate(self, image_name_list, board_size=(8,6)):
        ...
        ret, mtx, dist, rvecs, tvecs = \
        cv2.calibrateCamera(objpoints, imgpoints, \
                    gray.shape[::-1], None, None)
        self.mtx = mtx
        self.dist = dist
        ...

    def undistort(self, img):
        mtx = None
        return cv2.undistort(img, self.mtx, self.dist, None, mtx)

cam_1 = CAMERA()
cal_images = glob.glob("camera_cal/calibration*.jpg")
cam_1.calibrate(cal_images, (9,6))
img_undistorted = cam_1.undistort(img_ori)
```

![alt text][image1]
Fig. 1 The comparison of original raw image from camera and un-distorted image

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The un-distortion step is implemted in the method `pipeline()` of `IMAGE_PREPROCESSING` class, which implemented in `Part 2: On-line Processing and Pipeline / Integration: Combining all as IMAGE_PREPROCESSING` section. This is the first step of IMAGE_PREPROCESSING.pipeline(), using the method of CAMERA object created at first part. The `CAMERA.undistort()` use the cv2.undistort() and the `mtx` and `dist` calculated by `cv2.calibrateCamera` to inversely transform the image to generate a "flat" image.

To demonstrate this step, the code is simplyfy as follow, the full code can be checked in notebook.

```python
class CAMERA(object):
    def undistort(self, img):
        mtx = None
        return cv2.undistort(img, self.mtx, self.dist, None, mtx)

class IMAGE_PREPROCESSING(object):
    def __init__(self, camera_in):      
        self.camera = camera_in

    def pipeline(self, img_ori):
        # 1. Undistort the input image
        img_undistorted = self.camera.undistort(img_ori)

cam_1 = CAMERA()
img_preproc = IMAGE_PREPROCESSING(cam_1)
```

![alt text][image2]
Fig. 2 An un-distorted image of `./test_images/test4.jpg`

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

In this section, I constructed two pipe of steps to generate binary maskes for yellow lane line and white lane line respectively and combined these maskes by pixel-wise or operation.

These pipelines are implemented in `LANE_LINE_MASK` class, `pipeline()` method



```python
img_out = (bi_yellow | bi_white)
```

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3-1]
Fig. 3 Binary mask that extract yellow lines

![alt text][image3-2]
Fig. 4 Binary mask that extract white lines

![alt text][image3]
Fig. 5 Combined binary mask that contains both white and yellow lines

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

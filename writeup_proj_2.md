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
[image4]: ./output_images/warp/plot_straight_lines2_line.png "Original image with src Example"
[image4-1]: ./output_images/warp/plot_straight_lines2_birdeye.png "Warp Example"
[image4-2]: ./output_images/warp/plot_straight_lines2_img_trans_inverse_transe.png "Inversed Warp Example"
[image5-1]: ./output_images/histogram/plot_test4_histogram.png "Original histogram"
[image5-2]: ./output_images/histogram/plot_test4_histogram_weight.png "The weight"
[image5-3]: ./output_images/histogram/plot_test4_histogram_weighted.png "Weight histogram"
[image5-4]: ./output_images/full_pipeline/step/test4_full_img_lane_sliding.jpg "Result of sliding window search"
[image5-5]: ./output_images/full_pipeline/step/test4_full_img_lane_track.jpg "Result of tracking search"

[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

This file is the writeup for this project.

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The camera calibration step meants to find the intrinsic parameters and distortion coefficients of the camera we used. Because each camera is unique, and these parameters should be globally avelable for each camera, I create a class for cameras. The code of this step is writen in the notebook as a Python class called `CAMERA` in `Part 1: Off-line Preparation / Camera Calibration` section (in `code cell [2]`).
  
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

---

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The un-distortion step is implemted in the method `pipeline()` of `IMAGE_PREPROCESSING` class, which implemented in `Part 2: On-line Processing and Pipeline / Integration: Combining all as IMAGE_PREPROCESSING` section (in `code cell [14]`). This is the first step of IMAGE_PREPROCESSING.pipeline(), using the method of CAMERA object created at first part. The `CAMERA.undistort()` use the cv2.undistort() and the `mtx` and `dist` calculated by `cv2.calibrateCamera` to inversely transform the image to generate a "flat" image.

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

---

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

In this section, I constructed two pipe of steps to generate binary maskes for yellow lane line and white lane line respectively and combined these maskes by pixel-wise `or` operation. The yellow mask, white mask, and final result are shown in Fig. 3,4, and 5, respectively.

```python
img_out = (bi_yellow | bi_white)
```

These pipelines are implemented in `pipeline()` method, `LANE_LINE_MASK` class, in  `Part 2: On-line Processing and Pipeline / Image preprocessing / Step 2: Getting Binary Image of Lane-lines` section (in `code cell [7]`).

To generate the yellow-lane mask, I first convert the RGB color image into lnto HSV color space, then apply threshold to Hue and Saturation layer to get the hue in the range of [20-3, 20+2]. I shose HSV instead of HLS because the saturation is monotonically getting higher in HSV when the brightness is higher. I choose the saturation to be greater than 90, and doing pixel-wise `and` to generate the final yellow mask as shown in Fig. 3.

```python
bi_yellow = (H_binary & S_binary)
```

To generate the white mask, I do something more complicated. By observing the data, it seems a reasonable choise to pick up bright-enough pixels that both shown on sobel-x and sobel-y or the gradiant is high, so I do the following binary operation. The result is shown in Fig. 4.

```python
bi_white = (bi_mag | ( bi_sobel_x & bi_sobel_y)) & V_binary 
```


![alt text][image3-1]
Fig. 3 Binary mask that extract yellow lines

![alt text][image3-2]
Fig. 4 Binary mask that extract white lines

![alt text][image3]
Fig. 5 Combined binary mask that contains both white and yellow lines


---

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.


The perspective transform is implemented in `transform()` method, `IMAGE_WARPER` class, in  `Part 2: On-line Processing and Pipeline / Image preprocessing / Step 3: Warping Image to "Bird-eye View"` section (in `code cell [11]`).

This module has two jobs
- Perform perspective transform/inverse transform
- Provide the scale of meter/pixel of the warpped image

The important variable of this module is the transformation matrix `self.M_birdeye`. This matrix is calculated by `cv2.getPerspectiveTransform()` in `IMAGE_WARPER.cal_transform_matrices()`, which will be called in constructor of `IMAGE_WARPER`. To calculate the transformation matrix, I give the following source point (`self.warp_src`) and destination point (`self.warp_dst`) pairs.


```python
ori_x1 = 200
ori_x2 = img_size[0] - ori_x1
self.warp_src = \
    np.float32([ [593,450],\
                [686,450],\
                [ori_x2, img_size[1] ],\
                [ori_x1,img_size[1] ] ])
tranx_x1 = 300 # 350
trans_x2 = img_size[0] - tranx_x1
self.warp_dst = \
    np.float32([[tranx_x1,0],\
                [trans_x2,0],\
                [trans_x2, img_size[1]],\
                [tranx_x1, img_size[1]]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 593, 450      | 300, 0        | 
| 686, 450      | 980, 0        |
| 1080, 720     | 980, 720      |
| 200, 720      | 300, 720      |


The scale of meter/pixel of the warpped image is saved in `self.xm_per_pix` and `self.ym_per_pix`, whicj is measured and calculated according to the following fact
- The lane is 3.7 m in width
- The dash of lane line is 3 m long

AS teh result, the `self.xm_per_pix` and `self.ym_per_pix` are shown below
|   VAriable name   | value    |
|:-----------------:|:--------:| 
| `self.xm_per_pix` | 0.005355 |
| `self.ym_per_pix` |  0.04    |

, which, for example, has the following relation:

| Axis  |  pixel value | meter |
|:-----:|:------------:|:-----:|
| `x`   |  680         | 3.64  |
| `y`   |  720         | 28.8  |


I verified that my perspective transform was working as expected by drawing the `self.warp_src` points onto a test image. Then I generate the warpped image conuterpart and the inverse transformed image to verify that the lines appear parallel in the warped image.

![alt text][image4]

Fig. 6 Original image with `self.warp_src` drawn

![alt text][image4-1]

Fig. 7 Warped image

![alt text][image4-2]

Fig. 8 Inverse transform of the warped image



---

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In order to find the lane line pixel, I implemented a sophixticated class called `LANE_TRACKER` in `Part 2: On-line Processing and Pipeline / Lane-finding algorithm` section  (in `code cell [19]`) to perform the lane finding algorithms. 

The main entry of this module is the `LANE_TRACKER.pipeline()` method. The concept of this pipeline is explained as following.
1. Try finding lane using tracking method `LANE_TRACKER.search_around_poly()` to search the lines based on the previous result.
2. Calculate the curvature of lane and the position of vehicle 
3. Do some sanity check
4. If the check is fail, do step 1.~3. again but using sliding window method `LANE_TRACKER.fit_polynomial()`

**Sliding Window Search**

The sliding window search based on some select rectangle ROI to do local search of line points. The ROI start from the bottom of image, where the position is decided according to histogram of the lower-half image in x-direction. In each iteration, the ROI move up a step, the x-position is determined according the previous found points (average the position of points). 

It's mostly the same as one in lecture; however, I have done two modifications:
- Multiplying the histogram by weights that are higher at the center of image and lower at the boundary of image.
- Estimate the changing rate of sliding window in x-direction so that it will have higher possibility to find lane-line pixels in window of next y-level, according to the assumption of that the changing rate of the curvature of lane-line is small.

The following code block shows how I generate the weight, and a demonstration is shown in Fig. 9, 10, and 11. This piece of code is at the begining of `_find_lane_pixels()` in `LANE_TRACKER` class.

```python
# Weighted hidtogram, more weight at center
midpoint = np.int(histogram.shape[0]//2)
histogram_weight = np.array([midpoint - np.abs(midpoint - x) for x in range(len(histogram))] )
histogram = histogram * histogram_weight
```

![alt text][image5-1]

Fig. 9 Original histogream

![alt text][image5-2]

Fig. 10 The weight

![alt text][image5-3]

Fig. 11 Weighted histogram

The following codes shows how I estimate the changing rate of windows in x-direction. If there are enough points found in the current window, update the changing rate `leftx_delta` with an 1st-order linear filter. The new position of window will then be the mean of lane-line pixels in this iteration plus the changin-rate `leftx_delta`. If there are no enough lane-line points found, simply shift the current window with changing rate `leftx_delta`. The code piece can be found in `for window in range(nwindows):` section in `_find_lane_pixels()` of `LANE_TRACKER` class. The effect can be seen on Fig. 12 below.

```python
# Step through the windows one by one
for window in range(nwindows):
    ...
    # Left lane-line
    if len(good_left_inds) > minpix:
        leftx_current_new = int(np.mean(nonzerox[good_left_inds]) ) 
        leftx_delta += 0.2*(leftx_current_new - leftx_current)
        leftx_current = leftx_current_new + int(leftx_delta)
    else:
        leftx_current += int(leftx_delta)
    ...
```


**Tracking**

The tracking search use the previous found curves as the base line to generate a band of window as teh searching region for finding possible lane-line pixels. The function to do this calculation is writen in `search_around_poly()` of `LANE_TRACKER` class. The result is shown in Fig. 13. 


**Curve Fitting**

The final step is to fit left and right lane-line according the the pixels found in left and right window, repectively. However, the lane-line has a property of being parellel for every small line-segment, this property can be utilizd to make the result more robust. To utilize the property, I write a method called `parallelize_lines()` for `LANE_TRACKER` class, and to be called by `_fit_poly()` method. 

The concept is simple:
- Fit curves separately.
- Calculate the center curve by averaging the parameter, which has the same effect as averaging the x-value being calculated by both curve.
- Add offset on center curve to re-generate the left and right lanme-line curve.

The result of lane-line finding pipeline is shown in Fig. 12 and Fig. 13.

```python
def poly_func(self, poly_in, VAL_in, offset=0):
    """
    NOTE: VAL_in and VAL_out can be array or matrix.
    """
    return ( poly_in[0]*(VAL_in**2)\
                + poly_in[1]*VAL_in\
                + poly_in[2] + offset )

def _fit_poly(self, img_shape, leftx, lefty, rightx, righty):
    #Fit a second order polynomial to each with np.polyfit() 
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Parallelize two curves
    left_fit, right_fit, center_fit = \
        self.parallelize_lines(img_shape, left_fit, right_fit)    
    ...
    return left_fit, right_fit, left_fitx, right_fitx, ploty

def parallelize_lines(self, img_shape, left_fit, right_fit):
    # 1. Evaluate the offset of each line regarded to center line
    y_eval = float(img_shape[0] - 1) 
    center_fit = (left_fit + right_fit) * 0.5
    lx_center = self.poly_func(center_fit, y_eval)
    lx_left = self.poly_func(left_fit, y_eval, -lx_center)
    lx_right = self.poly_func(right_fit, y_eval, -lx_center)
    # Shift the center_fit
    left_fit = np.copy(center_fit )
    right_fit = np.copy(center_fit )
    left_fit[-1] += lx_left
    right_fit[-1] += lx_right
    return left_fit, right_fit, center_fit
```



![alt text][image5-4]

Fig. 12 Result of the sliding window search. Note that when there is less pixels found, the windows still moving toward a possible location according to the previous lane-line found.

![alt text][image5-5]
Fig. 13 Result of the tracking search

--

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

---

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

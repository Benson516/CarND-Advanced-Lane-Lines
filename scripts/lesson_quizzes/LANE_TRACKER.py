import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

class LANE_TRACKER(object):
    """
    """
    def __init__(self, prefix="lane_"):
        """
        """
        # Parameters
        #-------------------------#
        # Sliding window
        self.nwindows = 9 # Choose the number of sliding windows
        self.margin = 100 # 100 # Set the width of the windows +/- margin
        self.minpix = 50 # Set minimum number of pixels found to recenter window
        # Tracking
        self.track_margin = 70 # Set the width of the windows +/- margin
        self.track_minpix = 5000 # Set minimum number of pixels found to window
        #-------------------------#

        # Variables
        #-------------------------#
        self.prefix = prefix
        # Polynominal coefficients, in pixel
        self.left_fit = None
        self.right_fit = None
        # Polynominal coefficients, in meter
        self.left_fit_m = None
        self.right_fit_m = None
        # fitx for ploting
        self.left_fitx = None
        self.right_fitx = None
        #-------------------------#

    # Visualization
    #---------------------------------------------------------------------------------------------------#
    def _get_colored_line_point_image(self, binary_warped, leftx, lefty, rightx, righty, is_drawing_line_only=False):
        """
        """
        # Create an output image to draw on and visualize the result
        if is_drawing_line_only:
            out_img_l = np.zeros_like(binary_warped)
            out_img = np.dstack((out_img_l, out_img_l, out_img_l))
        else:
            out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]
        return out_img

    def _draw_strip_inplace(self, out_img, ploty, left_fitx, right_fitx, color=(0, 255, 0)):
        """
        """
        # Draw the lane
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        lane_window1 = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        lane_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        lane_pts = np.hstack((lane_window1, lane_window2))
        cv2.fillPoly(out_img, np.int_([lane_pts]), color)

    def _draw_strip(self, out_img, ploty, left_fitx, right_fitx, color=(0, 255, 0)):
        """
        """
        # Draw the lane
        lane_img = np.zeros_like(out_img)
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        lane_window1 = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        lane_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        lane_pts = np.hstack((lane_window1, lane_window2))
        cv2.fillPoly(lane_img, np.int_([lane_pts]), color)
        return cv2.addWeighted(out_img, 1, lane_img, 0.5, 0)

    def _draw_polyline_inplace(self, out_img, poly_in, color=(255,0,255), thickness=5):
        """
        """
        # Generate x and y values for plotting
        ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
        plotx = self.poly_func(poly_in, ploty)
        #
        line_pts = np.array([np.transpose(np.vstack([plotx, ploty]))])
        cv2.polylines(out_img, np.int_([line_pts]), isClosed=False, color=color, thickness=thickness)



    #---------------------------------------------------------------------------------------------------#
    # end Visualization

    def trans_poly_pixel_2_meter(self, poly_in, m_per_pix_out, m_per_pix_in):
        """
        """
        deg = len(poly_in) - 1
        # print("poly_in = %s" % str(poly_in) )
        poly_out =  np.array( [ m_per_pix_out * poly_in[idx]/( m_per_pix_in**(deg-idx) ) for idx in range(len(poly_in))] )
        # print("poly_out = %s" % str(poly_out) )
        return poly_out

    def poly_func(self, poly_in, VAL_in, offset=0):
        """
        NOTE: VAL_in and VAL_out can be array or matrix.
        VAL_out = poly_in[0]*(VAL_in**2) + poly_in[1]*VAL_in + poly_in[2] + offset
        """
        return ( poly_in[0]*(VAL_in**2) + poly_in[1]*VAL_in + poly_in[2] + offset )

    def curvature_func(self, poly_in, VAL_in, is_abs=False):
        """
        NOTE: VAL_in and VAL_out can be array or matrix.
        _R = (1.0 + (2.0*poly_in[0]*VAL_in + poly_in[1])**2)**(1.5)/abs(2.0*poly_in[0])
        """
        if is_abs:
            return ( (1.0 + (2.0*poly_in[0]*VAL_in + poly_in[1])**2)**(1.5)/np.absolute(2.0*poly_in[0]) )
        else:
            return ( (1.0 + (2.0*poly_in[0]*VAL_in + poly_in[1])**2)**(1.5)/(2.0*poly_in[0]) )

    def _fit_poly(self, img_shape, leftx, lefty, rightx, righty):
        ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        #
        # Parallelize two curves
        left_fit, right_fit, center_fit = self._parallelize_lines(img_shape, left_fit, right_fit)

        # Generate x and y values for plotting
        ploty = np.linspace(0, img_shape[0]-1, img_shape[0])

        ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
        left_fitx = self.poly_func(left_fit, ploty)
        right_fitx = self.poly_func(right_fit, ploty)

        return left_fit, right_fit, left_fitx, right_fitx, ploty

    def _parallelize_lines(self, img_shape, left_fit, right_fit):
        # 1. Evaluate the offset of each line regarded to center line
        y_eval = float(img_shape[0] - 1)
        center_fit = (left_fit + right_fit) * 0.5
        lx_center = self.poly_func(center_fit, y_eval)
        lx_left_delta = self.poly_func(left_fit, y_eval, -lx_center)
        lx_right_delta = self.poly_func(right_fit, y_eval, -lx_center)
        # Shift the center_fit
        left_fit = np.copy(center_fit )
        right_fit = np.copy(center_fit )
        left_fit[-1] += lx_left_delta
        right_fit[-1] += lx_right_delta
        return left_fit, right_fit, center_fit

    def _update_poly(self, left_fit, right_fit):
        # Update
        self.left_fit = left_fit
        self.right_fit = right_fit

    def _update_fitx(self, left_fitx, right_fitx):
        # Update
        self.left_fitx = left_fitx
        self.right_fitx = right_fitx

    def _find_lane_pixels(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

        # Weighted hidtogram, more weight at center
        midpoint = np.int(histogram.shape[0]//2)
        histogram_weight = np.array([midpoint - np.abs(midpoint - x) for x in range(len(histogram))] )
        histogram = histogram * histogram_weight

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        endpoint_left = midpoint - 50
        endpoint_right = midpoint + 50
        leftx_base = (endpoint_left-1) - np.argmax(histogram[(endpoint_left-1)::-1]) # Search from the center, the np.argmax() will only return the first found
        rightx_base = np.argmax(histogram[endpoint_right:]) + endpoint_right

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = self.nwindows
        # Set the width of the windows +/- margin
        margin = self.margin
        # Set minimum number of pixels found to recenter window
        minpix = self.minpix

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # The increament of window
        leftx_delta = 0
        rightx_delta = 0

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        win_points_list = []
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            ### TO-DO: Find the four below boundaries of the window ###
            win_xleft_low = leftx_current - margin  # Update this
            win_xleft_high = leftx_current + margin  # Update this
            win_xright_low = rightx_current - margin  # Update this
            win_xright_high = rightx_current + margin  # Update this

            # Save the window points for ploting
            # Note: (left_rectangle_corner_1, left_rectangle_corner_2, right_rectangle_corner_1, right_rectangle_corner_2)
            win_points_list.append( ( (win_xleft_low,win_y_low), (win_xleft_high,win_y_high), (win_xright_low,win_y_low), (win_xright_high,win_y_high)) )

            ### TO-DO: Identify the nonzero pixels in x and y within the window ###
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            ### TO-DO: If you found > minpix pixels, recenter next window ###
            ### (`right` or `leftx_current`) on their mean position ###
            # Left lane-line
            if len(good_left_inds) > minpix:
                leftx_current_new = int(np.mean(nonzerox[good_left_inds]) )
                leftx_delta += 0.2*(leftx_current_new - leftx_current)
                leftx_current = leftx_current_new + int(leftx_delta)
            else:
                leftx_current += int(leftx_delta)
            # Right lane-line
            if len(good_right_inds) > minpix:
                rightx_current_new = int(np.mean(nonzerox[good_right_inds]) )
                rightx_delta += 0.2*(rightx_current_new - rightx_current)
                rightx_current = rightx_current_new + int(rightx_delta)
            else:
                rightx_current += int(rightx_delta)
            # pass # Remove this when you add your function

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, win_points_list # out_img

    def fit_polynomial(self, binary_warped, debug=False, verbose=False):
        # Find our lane pixels first
        leftx, lefty, rightx, righty, win_points_list = self._find_lane_pixels(binary_warped)

        # Fit new polynomials
        left_fit, right_fit, left_fitx, right_fitx, ploty = self._fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
        # Update
        self._update_poly(left_fit, right_fit)
        self._update_fitx(left_fitx, right_fitx)

        ## Visualization ##
        #----------------------------#
        # Generate the output image with lane-line pixels marked
        out_img = self._get_colored_line_point_image(binary_warped, leftx, lefty, rightx, righty)
        # Draw the lane
        out_img = self._draw_strip(out_img, ploty, left_fitx, right_fitx)

        if debug:
            # Draw the windows on the visualization image
            for win_points in win_points_list:
                cv2.rectangle(out_img, win_points[0], win_points[1], (128,128,0), 2)
                cv2.rectangle(out_img, win_points[2], win_points[3], (128,128,0), 2)

        # # Plots the left and right polynomials on the lane lines
        self._draw_polyline_inplace(out_img, self.left_fit)
        self._draw_polyline_inplace(out_img, self.right_fit)
        # Draw the center line
        self._draw_polyline_inplace(out_img, (self.left_fit + self.right_fit)*0.5 )

        #----------------------------#
        ## End visualization steps ##

        return out_img

    def search_around_poly(self, binary_warped, debug=False, verbose=False):
        #
        # Save the old fitx for ploting the searching region
        left_fitx_old = self.left_fitx
        right_fitx_old = self.right_fitx
        #
        margin = self.track_margin
        left_fit = self.left_fit
        right_fit = self.right_fit

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        ### TO-DO: Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        ### Hint: consider the window areas for the similarly named variables ###
        ### in the previous quiz, but change the windows to our new search area ###
        left_lane_inds = ((nonzerox > self.poly_func(left_fit, nonzeroy, -margin) )
                          & (nonzerox < self.poly_func(left_fit, nonzeroy, margin)) )
        right_lane_inds = ((nonzerox > self.poly_func(right_fit, nonzeroy, -margin) )
                           & (nonzerox < self.poly_func(right_fit, nonzeroy, margin)) )

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Check if the points in region is not plenty enough
        if len(leftx) < self.track_minpix or len(rightx) < self.track_minpix:
            print("Reset track, len(leftx) = %d, len(rightx) = %d" % (len(leftx), len(rightx)))
            return None

        # Fit new polynomials
        left_fit, right_fit, left_fitx, right_fitx, ploty = self._fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
        # Update
        self._update_poly(left_fit, right_fit)
        self._update_fitx(left_fitx, right_fitx)

        ## Visualization ##
        #----------------------------#
        # Generate the output image with lane-line pixels marked
        out_img = self._get_colored_line_point_image(binary_warped, leftx, lefty, rightx, righty)
        # Draw the lane
        out_img = self._draw_strip(out_img, ploty, left_fitx, right_fitx)

        if debug:
            # Draw windows
            window_img = np.zeros_like(out_img)
            self._draw_strip_inplace(window_img, ploty, left_fitx_old-margin, left_fitx_old+margin, color=(128,128,0))
            self._draw_strip_inplace(window_img, ploty, right_fitx_old-margin, right_fitx_old+margin, color=(128,128,0))
            out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        # # Plot the polynomial lines onto the image
        self._draw_polyline_inplace(out_img, self.left_fit)
        self._draw_polyline_inplace(out_img, self.right_fit)
        # Draw the center line
        self._draw_polyline_inplace(out_img, (self.left_fit + self.right_fit)*0.5 )
        #----------------------------#
        ## End visualization steps ##

        return out_img

    def find_lane(self, binary_warped, is_fixed_to_sliding_window=False, debug=False):
        """
        """
        is_tracking = False
        if (not is_fixed_to_sliding_window) and ( (not self.left_fit is None) and (not self.right_fit is None) ):
            # print("track")
            out_img = self.search_around_poly(binary_warped, debug=debug)
            if not out_img is None:
                is_tracking = True
                return out_img, is_tracking
        # print("sliding window")
        out_img = self.fit_polynomial(binary_warped, debug=debug)
        return out_img, is_tracking




    def calculate_radius_and_offset(self, binary_warped, xm_per_pix, ym_per_pix, verbose=False):
        # 1. Calculate curvature
        y_eval_m = float(binary_warped.shape[0] - 1) * ym_per_pix
        self.left_fit_m = self.trans_poly_pixel_2_meter( self.left_fit, xm_per_pix, ym_per_pix)
        self.right_fit_m = self.trans_poly_pixel_2_meter( self.right_fit, xm_per_pix, ym_per_pix)
        R_left = self.curvature_func(self.left_fit_m, y_eval_m)
        R_right = self.curvature_func(self.right_fit_m, y_eval_m)
        R_avg = self.curvature_func( (self.left_fit_m + self.right_fit_m )*0.5, y_eval_m)
        R_dict = dict()
        R_dict["R_left"] = R_left
        R_dict["R_right"] = R_right
        R_dict["R_avg"] = R_avg

        # 2. Calculate the vehicle position with respect to center
        x_img_center = float(binary_warped.shape[1] - 1) * xm_per_pix * 0.5
        lx_left = self.poly_func(self.left_fit_m, y_eval_m, -x_img_center)
        lx_right = self.poly_func(self.right_fit_m, y_eval_m, -x_img_center)
        lx_avg = (lx_left + lx_right)*0.5
        lx_delta = lx_right - lx_left
        lx_dict = dict()
        lx_dict["lx_left"] = lx_left
        lx_dict["lx_right"] = lx_right
        lx_dict["lx_avg"] = lx_avg
        lx_dict["lx_delta"] = lx_delta

        if verbose:
            log_out = ""
            log_out += "(R_left, R_right, R_avg) = (%f, %f, %f)" % (R_left, R_right, R_avg)
            log_out += "\n"
            log_out += "(lx_left, lx_right, lx_avg, lx_delta) = (%f, %f, %f, %f)" % (lx_left, lx_right, lx_avg, lx_delta)
            print(log_out)

        return R_dict, lx_dict

    def sanity_check(self, R_dict, lx_dict):
        if lx_dict["lx_delta"] > 4.2 or lx_dict["lx_delta"] < 3.2:
            print("[check] too close")
            return False
        if ( R_dict["R_left"]*R_dict["R_right"] < 0.0) and abs(R_dict["R_left"] - R_dict["R_left"]) > 200.0:
            print("[check] Different direction")
            return False
        if abs(R_dict["R_left"]) < 100.0 or abs(R_dict["R_right"]) < 100.0:
            print("[check] Impossible curvature")
            return False
        return True

    def pipeline(self, binary_warped, xm_per_pix, ym_per_pix, debug=True, verbose=False):
        """
        """
        # 1.Find lane
        out_img, is_tracking = self.find_lane(binary_warped, is_fixed_to_sliding_window=False, debug=debug)
        # 2. Calculate curvature and the vehicle position with respect to center
        R_dict, lx_dict = self.calculate_radius_and_offset(binary_warped, xm_per_pix, ym_per_pix, verbose=verbose)
        # 3. Sanity check
        is_passed = self.sanity_check( R_dict, lx_dict)
        if (not is_passed) and is_tracking:
            print("Reset track")
            # 1. Do it again with sliding window
            out_img, is_tracking = self.find_lane(binary_warped, is_fixed_to_sliding_window=True, debug=debug)
            # 2. Calculate curvature and the vehicle position with respect to center
            R_dict, lx_dict = self.calculate_radius_and_offset(binary_warped, xm_per_pix, ym_per_pix, verbose=verbose)
            # 3. Sanity check
            is_passed = self.sanity_check( R_dict, lx_dict )

        return out_img, R_dict, lx_dict, is_tracking

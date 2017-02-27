"""
This script include functions for the car center-front camera to estimate
road curvature from road lines.
"""
import cv2
import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

from params import *

"""

FUNCTIONS FOR IMAGE CALIBRATION AND UNDISTORTION

"""

def cal_undistort(img, objpoints, imgpoints):
    """
    Generates undistorted image
    :param img: array with distorted image
    :param objpoints: list with object points
    :param imgpoints: list with image points
    :return: array with undistorted image
    """
    img_size = (img.shape[0], img.shape[1])
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    return cv2.undistort(img, mtx, dist, None, mtx)


def calibrate_camera(path_to_images, nx, ny):
    """
    Load distorted images to calculate object points and image points
    :param path_to_image: string with image path
    :param nx: number of inside corners in x
    :param ny: number of inside corners in y
    :return: couple of lists with object points and image points
    """
    # Read calibration images from source path:
    images = glob.glob(path_to_images + 'calibration*')

    # Arrays to store object points and image points from all images:
    objpoints = []
    imgpoints = []

    # Prepare object points (3D points in real world):
    objp = np.zeros((ny*nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    for fname in images:
        # read image:
        img = mpimg.imread(fname)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        #
        # If found, draw corners
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
        # Read in an image
    return objpoints, imgpoints

"""

FUNCTIONS FOR IMAGE BINARIZATION FROM GRADIENTS AND COLOR TRANSFORMATIONS

"""

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    """
    Generates a binary image with Sobel transformation
    :param img: array with input image to transform
    :param orient: string of values x or y defining the Sobel derivative direction
    :param sobel_kernel: int size of Sobel kernel
    :param thresh: tuple with thresholds for the gradient magnitude in the selected direction
    :return: array with binarized image
    """
    assert sobel_kernel % 2, "sobel_kernel nees to be odd"

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

def mag_sobel_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    """
    Generates a binary image using the Sobel module in the x and y directions
    :param img: array with input image to transform
    :param sobel_kernel: int size of Sobel kernel
    :param thresh: tuple with thresholds for the gradient magnitude
    :return: array with binarized image
    """
    assert sobel_kernel % 2, "sobel_kernel nees to be odd"

    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    abs_sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    abs_sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    abs_sobelxy = (abs_sobelx**2 + abs_sobely**2)**0.5
    scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return binary_output

def dir_sobel_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    """
    Generates a binary image using the Sobel direction angles
    :param img: array with input image to transform
    :param sobel_kernel: int size of Sobel kernel
    :param thresh: tuple with thresholds for the gradient angle
    :return: array with binarized image
    """
    assert sobel_kernel % 2, "sobel_kernel nees to be odd"

    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    abs_sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    abs_sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    absgraddir = np.arctan2(abs_sobely, abs_sobelx)
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output

def combine_sobel_trans(image, ksize, thresh=(20, 200), thres_angles = (0, np.pi / 2)):
    """
    Combine all Sobel transformations into a single binarized image
    :param image: array with imput image
    :param ksize: int with size of Sobel kernel
    :param thres: tuple with thresholds for the gradients
    :param thres_angles: tuple with thresholds (rads) for the gradient angle
    :return:
    """
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=thresh)
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=thresh)
    mag_binary = mag_sobel_thresh(image, sobel_kernel=ksize, mag_thresh=thresh)
    dir_binary = dir_sobel_threshold(image, sobel_kernel=ksize, thresh=thres_angles)

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    return combined


def s_select(img, thresh=(0, 255)):
    """
    Convert RGB image to HLS, and returns a binarized image using the S channel
    and defining a threshold
    :param img: array with image
    :param thresh: tuple with thresholds
    :return: binarized image from the S channel and applying a threshold
    """
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    S = hls[:, :, 2]
    binary_output = np.zeros_like(S)
    binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1
    return binary_output


def channel_select(img, channel):
    """
    Convert RGB image to HLS, and returns a binarized image for the selected channel
    :param img: array with image
    :param channel: int for channel selection: 0 = H; 1 = L; 2 = S
    :return: binarized image for the selected channel
    """
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    return hls[:, :, channel]


def color_gradient_pipeline(img, s_thresh=(170, 255), sobel_thres=(0, 250)):
    """
    Perform Gradient and color transformations over input image
    :param img: array image
    :param s_thresh: color threshold for the S channel
    :param sobel_thres: color threshold for the Sobel color threshold
    :return: array with transformed image
    """
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    l_channel = channel_select(img, 1)
    s_binary = s_select(img, thresh=s_thresh)
    # Sobel angle
    sxbinary = mag_sobel_thresh(l_channel, sobel_kernel=3, mag_thresh=sobel_thres)
    # Stack each channel
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary

"""

FUNCTION TO CHANGE IMAGE PERSPECTIVE

"""


def perspective_transformation(img, src, dst, inverse=False):
    """
    Performs image perspective transformation
    :param img: array with original image
    :param src: list with 4 points in the input image
    :param dst: list with the corresponding 4 points represented on the output image
    :param inverse: Boolean to perform inverse or not transformation
    :return: array with the image transformed
    """
    if inverse:
        M = cv2.getPerspectiveTransform(dst, src)
    else:
        M = cv2.getPerspectiveTransform(src, dst)
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped

"""

FUNCTION TO IDENTIFY LINES

"""


def identify_lines(img, leftx_prev=None, rightx_prev=None):
    """
    Identify road lines pixels and fits them with a curve
    :param img: array with input binary image
    :param leftx_prev: previous image left line x position
    :param rightx_prev: previous image right line x position
    :return out_img: array with output binary image with lines drawn
    :return nonzerox: x positions of all nonzero pixels in the image
    :return nonzeroy: y positions of all nonzero pixels in the image
    :return left_lane_inds: nonzero pixels in x within the search windows
    :return right_lane_inds: nonzero pixels in y within the search windows
    :return left_fit: function with the left line fit
    :return right_fit: function with the right line fit
    :return leftx_current: last window on their mean position for left line
    :return rightx_current: window on their mean position for right line
    :return leftx: left line x pixels positions
    :return rightx: right line x pixels positions
    :return lefty:  left line y pixels positions
    :return righty: right line y pixels positions
    """
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((img, img, img)) * 255

    if leftx_prev is None or rightx_prev is None:
        # Take a histogram of the bottom half of the image
        histogram = np.sum(img[int(img.shape[0]*0.35):, :], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_current = np.argmax(histogram[:midpoint])
        rightx_current = np.argmax(histogram[midpoint:]) + midpoint
    else:
        leftx_current = leftx_prev
        rightx_current = rightx_prev

    # Set height of windows
    window_height = np.int(img.shape[0] / NWINDOWS)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(NWINDOWS):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
        nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
        nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # plt.imshow(out_img)
    # plt.savefig('/home/carnd/Self-Driving-Car-ND/Term1/P4 - Advance Traffic Lines Recognition/images/test2.png')
    return out_img, nonzerox, nonzeroy, left_lane_inds, right_lane_inds, left_fit, right_fit, \
           leftx_current, rightx_current, leftx, rightx, lefty, righty


"""

FUNCTION TO CALCULATE LINES CURVATURES AND CAR POSITION

"""


def measure_curvature(y_eval, lefty, righty, leftx, rightx, leftx_current, rightx_current,
                      img_shape):
    """
    Calculates left and right lines curvatures in meters
    :param y_eval: y-value where it is calculated the radius of curvature
    :param lefty: left line y pixels positions
    :param righty: right line y pixels positions
    :param leftx: left line x pixels positions
    :param rightx: right line x pixels positions
    :param leftx_current: last window on their mean position for left line
    :param rightx_current: window on their mean position for right line
    :param img_shape: image shape
    :return: left and right lines curvatures and offset from center in meters
    """
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty * YM_PER_PIX, leftx * XM_PER_PIX, 2)
    right_fit_cr = np.polyfit(righty * YM_PER_PIX, rightx * XM_PER_PIX, 2)

    # Calculate the new radius of curvature in meters
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * YM_PER_PIX + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * YM_PER_PIX + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    # Calculate position with respect to center:
    img_midpoint = img_shape[1] / 2
    car_position = (leftx_current + rightx_current) / 2
    displaced_from_center = (img_midpoint - car_position) * XM_PER_PIX

    return round(left_curverad, 2), round(right_curverad, 2), round(displaced_from_center, 2)


def draw_lines_real(undist, warped, left_fit, right_fit, src, dst, left_curverad, right_curverad, displaced_from_center):
    """
    Draw polygon calculated from line analysis over original image
    and add informative text with road curvature and distance from center
    :param undist: array with original image
    :param warped: array with warped binary image
    :param left_fit: left road line fit
    :param right_fit: right road line fit
    :param src: vertice points in the original image
    :param dst: vertice points in the proyected image
    :param left_curverad: curvature left line (m)
    :param right_curverad: curvature right line (m)
    :param displaced_from_center: displacement from line (m)
    :return: array with original image +  added information
    """
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Fit a second order polynomial to pixel positions in each fake lane line
    ploty = np.linspace(0, color_warp.shape[0] - 1, color_warp.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = perspective_transformation(color_warp, np.float32(src), np.float32(dst), inverse=True)

    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    # add text:
    radii = round(np.mean([left_curverad, right_curverad]), 2)
    if radii > RADII_TRESHOLD:
        radii = 'INF'

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, 'Current Curvature: {0} m '.format(radii), (100, 90), font, 1, (200, 255, 155), 2, cv2.LINE_AA)
    cv2.putText(result, 'Position in line: {0} m '.format(displaced_from_center), (100, 130), font, 1, (200, 255, 155), 2, cv2.LINE_AA)
    # plt.imshow(result)
    # plt.savefig('/home/carnd/Self-Driving-Car-ND/Term1/P4 - Advance Traffic Lines Recognition/images/test.png')
    return result


def real_time_processing(img):
    img_transformed = color_gradient_pipeline(img, s_thresh=(170, 255), sobel_thres=(40, 250))
    img_transformed = perspective_transformation(img_transformed, np.float32(VERTICES), np.float32(VERTICES_TRANSFORMED))

    out_img, nonzerox, nonzeroy, left_lane_inds, right_lane_inds, left_fit, right_fit, \
               leftx_current, rightx_current, leftx, rightx, lefty, righty = identify_lines(img_transformed, NWINDOWS)

    ploty = np.linspace(0, img_transformed.shape[0] - 1, img_transformed.shape[0])
    y_eval = np.max(ploty)
    left_curverad, right_curverad, displaced_from_center = measure_curvature(y_eval, lefty, righty, leftx, rightx, leftx_current, rightx_current, img_transformed.shape)
    return draw_lines_real(img, img_transformed, left_fit, right_fit, VERTICES, VERTICES_TRANSFORMED, left_curverad, right_curverad, displaced_from_center)


def create_video(path_to_video):
    from moviepy.editor import VideoFileClip
    clip2 = VideoFileClip(path_to_video)
    challenge_clip = clip2.fl_image(real_time_processing)
    challenge_clip.write_videofile(path_to_video.replace(".mp4", "_solved.mp4"), audio=False)



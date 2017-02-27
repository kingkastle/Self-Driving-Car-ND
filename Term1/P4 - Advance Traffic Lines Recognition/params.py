"""
This script includes all parameters used
"""
import numpy as np

# Vertices in original image:
VERTICES = np.array([[200, 680], [570, 460], [710, 460], [1100, 680]], dtype=np.int32)  # this step requires manual fine-tune

# Vertices in transformed image:
VERTICES_TRANSFORMED = np.array([[100, 700], [100, 150], [1100, 150], [1100, 700]], dtype=np.int32)

NWINDOWS = 9

# Define conversions in x and y from pixels space to meters
YM_PER_PIX = 21/720  # meters per pixel in y dimension
XM_PER_PIX = 3.7/1000  # meters per pixel in x dimension

# radii_threshold:
RADII_TRESHOLD = 10000

# Path to example files:
PATH_TO_EXAMPLE_FILES = 'CarND-Advanced-Lane-Lines/test_images/'

# Path to example files:
PATH_TO_CAMERA_CAL = 'CarND-Advanced-Lane-Lines/camera_cal/'

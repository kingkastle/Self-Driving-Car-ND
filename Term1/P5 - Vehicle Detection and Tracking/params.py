# PATH TO TEST FILES:
PATH_TEST = '/home/carnd/Self-Driving-Car-ND/Term1/P5 - Vehicle Detection and Tracking/CarND-Vehicle-Detection/test_images/'

# PATH TO OUTPUT IMAGES:
OUTPUT_IMAGES = '/home/carnd/Self-Driving-Car-ND/Term1/P5 - Vehicle Detection and Tracking/images/'

# HOG Parameters
PIX_PER_CELL = 8
CELL_PER_BLOCK = 2
ORIENT = 9
HOG_CHANNEL = 'ALL'

# COLOR parameters:
COLOR_SPACE = 'YCrCb'
SPATIAL_SIZE = (16, 16)
HIST_BINS = 32
HIST_RANGE = (0, 256)

# Transformations:
SPATIAL_FEAT = True
HIST_FEAT = True
HOG_FEAT = True

# DATASET path:
DATASET_PATH = '/home/carnd/Self-Driving-Car-ND/Term1/P5 - Vehicle Detection and Tracking/dataset/'

# MODEL training:
SCORE_METRIC = 'f1'
RANDOM_STATE = 42
PORTION = 100

# SLIDING Window:
X_START_STOP = [200, None]
Y_START_STOP = [370, None]
XY_WINDOW = (130, 130)
XY_OVERLAP = (0.75, 0.75)

# BOXES colors:
COLOR = (0, 0, 255)
THICK = 6

# HEATMAP threshold:
THRESHOLD = 10
import numpy as np
import cv2
from skimage.feature import hog
import matplotlib

matplotlib.use('AGG')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage.measurements import label
import os
from glob import glob
from params import *
import pickle
from time import time
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings("ignore")


# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    """
    For each of the image channels it is calculated its distribution
    :param img: array with the image
    :param nbins: number of bins to use in the histogram
    :param bins_range: tuple with range of the bins (correspond to pixels)
    :return: list with colors distributions
    """
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    ghist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    bhist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges) - 1]) / 2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# IMG reduction + color transformation:
def bin_spatial(img, color_space='RGB', size=(32, 32)):
    """
    Transform image to a different color space and perform resizing
    :param img: array with image
    :param color_space: Final color space to use
    :param size: size of the final image
    :return: features from image and image in the new color space
    """
    # Convert image to new color space (if specified)
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(feature_image, size).ravel()
    # Return the feature vector
    return features, feature_image


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    """
    Perform Hog transformation
    :param img: array to image
    :param orient: number of different orientations to consider
    :param pix_per_cell: number of pixels included per cell
    :param cell_per_block: number of cells included per block
    :param vis: Boolean to generate HOG visualization
    :param feature_vec: Boolean
    :return:
    """
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
                       visualise=False, feature_vector=feature_vec)
        return features


def extract_feats_dataset():
    """
    Process all images in the dataset and generates a pickle object with the associated features and labels arrays.
    Features are distributed in two different folders: vehicles
    :param dataset_path: path to dataset and non-vehicles
    :return: features and dict sizes pickle object
    """
    # Create a list to append feature vectors to
    features = {'vehicles': {}, 'non-vehicles': {}}
    sizes_sources = {'vehicles': {}, 'non-vehicles': {}}
    for label in ['vehicles', 'non-vehicles']:
        for dir, subdir, files in os.walk(DATASET_PATH + label):
            if len(subdir) > 0:
                for source in subdir:
                    features[label][source] = []
            else:
                source = dir[dir.rfind("/") + 1:]
                # features[label][source] = [x for x in files if ".png" in x]
                features[label][source] = [single_img_features(mpimg.imread(os.path.join(dir, x))) for x in files if
                                           ".png" in x]
                sizes_sources[label][source] = len(features[label][source])
                # save results to pandas dataframe:
                pickle.dump(features, open(DATASET_PATH + "features.p", "wb"))
    return features, sizes_sources


def create_sets(features, sizes_sources, split=[0.7, 0.2, 0.1]):
    """
    Generate train/test/validation sets and corresponding label sets
    :param features: json with image files
    :param sizes_sources: json with vehicles/non-vehicles dimensions
    :param split: splits portions for train/test/validation sets
    :return: numpy arrays with train/test/validation sets and corresponding labels
    """
    feat_train, label_train, feat_test, label_test, feat_val, label_val = [], [], [], [], [], []
    for label in ['vehicles', 'non-vehicles']:
        for source in features[label].keys():
            idx_items_train = int(split[0] * sizes_sources[label][source])
            idx_items_test = int(np.sum(split[:2]) * sizes_sources[label][source])
            feat_train.extend(features[label][source][:idx_items_train])
            feat_test.extend(features[label][source][idx_items_train:idx_items_test])
            feat_val.extend(features[label][source][idx_items_test:])
            if label == 'vehicles':
                label_value = 1
            else:
                label_value = 0
            label_train.extend([label_value] * (len(features[label][source][:idx_items_train])))
            label_test.extend([label_value] * (len(features[label][source][idx_items_train:idx_items_test])))
            label_val.extend([label_value] * (len(features[label][source][idx_items_test:])))

    return np.array(feat_train), np.array(label_train), np.array(feat_test), np.array(label_test), np.array(
        feat_val), np.array(label_val)


def single_img_features(img, spatial_feat=True, hist_feat=True, hog_feat=True):
    """
        From a list of images it calculates the associated features (color and gradient)
        :param img: image array
        :param spatial_feat: Boolean to perform spatial features or not
        :param hist_feat: Boolean to perform color features or not
        :param hog_feat: Boolean to perform HOG features or not
        :return: features associated to the corresponding images
    """
    # 1) Define an empty list to receive features
    img_features = []
    # 2) Apply color conversion if other than 'RGB'
    spatial_features, feature_image = bin_spatial(img, color_space=COLOR_SPACE, size=SPATIAL_SIZE)
    # 3) Compute spatial features if flag is set
    if spatial_feat == True:
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=HIST_BINS)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog_feat == True:
        if HOG_CHANNEL == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                     ORIENT, PIX_PER_CELL, CELL_PER_BLOCK,
                                                     vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, HOG_CHANNEL], ORIENT,
                                            PIX_PER_CELL, CELL_PER_BLOCK, vis=False, feature_vec=True)
        # 8) Append features to list
        img_features.append(hog_features)

    # 9) Return concatenated array of features
    return np.concatenate(img_features)


def run_models(feat_train, label_train, feat_test, label_test):
    """
    Test different model optimizations
    :param feat_train: numpy array features train
    :param label_train: numpy array labels train
    :param feat_test: numpy array features test
    :param label_test: numpy array labels test
    :return: best performing model
    """
    best_f1 = 0
    for num_variables in range(10, 100, 30):
        for C in [1, 10]:
            for tol in [1e-3, 1e-5, 1e-4]:
                pipeline = Pipeline([('scaler', StandardScaler()), ('selection', SelectPercentile()),
                                     ('clf', LinearSVC(random_state=RANDOM_STATE, C=C, tol=tol))])
                pipeline.fit(feat_train, label_train)
                print("Results for Percentile: {0}, C: {1}, tol: {2}:".format(num_variables, C, tol))
                y_pred = pipeline.predict(feat_test)
                print(classification_report(label_test, y_pred, target_names=['non-car', 'car']))
                current_f1 = metrics.f1_score(label_test, y_pred)
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    # save results to pandas dataframe:
                    pickle.dump(pipeline, open(DATASET_PATH + "estimator.p", "wb"))


# def extract_feats_dataset():
#     """
#     Process all images in the dataset and generates a pickle object with the associated features and labels arrays.
#     Features are distributed in two different folders: vehicles
#     :param dataset_path: path to dataset and non-vehicles
#     :return: features pickle object
#     """
#     # Create a list to append feature vectors to
#     features = {'vehicles': {}, 'non-vehicles': {}}
#     for label in ['vehicles', 'non-vehicles']:
#         for dir, subdir, files in os.walk(DATASET_PATH + label):
#             if len(subdir) > 0:
#                 for source in subdir:
#                     features[label][source] = []
#             else:
#                 source = dir[dir.rfind("/")+1:]
#                 features[label][source] = [single_img_features(mpimg.imread(os.path.join(dir, x))) for x in files if ".png" in x]
#     return features


def train_model(features_train, labels_train, pipeline, parameters, n_iter_search, cv):
    """
    This function uses a SearchCV object to identify the best performing classifier
    :param features_train: feature set used for training
    :param labels_train: labels used for training
    :param pipeline: pipeline for the corresponding classifier
    :param parameters: pipelines parameters to tune
    :param n_iter_search: % of parameters combinations to evaluate
    :param cv: cross-validation method
    :return: best trained estimator, best parameters, best score and processing time
    """
    t0 = time()
    # Define GridCV Object:
    grid_search = RandomizedSearchCV(pipeline, parameters, scoring=SCORE_METRIC, n_jobs=-1, cv=cv, verbose=1,
                                     n_iter=n_iter_search)
    grid_search.fit(features_train, labels_train)
    estimator = grid_search.best_estimator_
    best_parameters = grid_search.best_estimator_.get_params()
    best_score = grid_search.best_score_

    # save results to pandas dataframe:
    pickle.dump(estimator, open(DATASET_PATH + "estimator.p", "wb"))
    return estimator, best_parameters, best_score, time() - t0


def search_portion(percent, parameters):
    """
    Retrieve the number of iterations to cover a percent of the parameters total space
    :param percent: float between 0 and 1 to define the search portion during tuning
    :param parameters: dict with pipeline parameters
    :return: int with the number of points in feature space
    """
    states = 1
    for element in parameters.keys():
        states *= len(parameters[element])
    return int(percent * states / 100.)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, score='f1',
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), test_score=None):
    """
    Generate a simple plot of the test and training learning curve.
    :param estimator: object type that implements the "fit" and "predict" methods An object of that type which is
    cloned for each validation.
    :param title: string Title for the chart.
    :param X: array-like, shape (n_samples, n_features). Training vector, where n_samples is the number of samples and
    n_features is the number of features.
    :param y: array-like, shape (n_samples) or (n_samples, n_features), optional
    Target relative to X for classification or regression; None for unsupervised learning.
    :param ylim: tuple, shape (ymin, ymax), optional. Defines minimum and maximum yvalues plotted.
    :param cv: int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        `StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.
    :param score: Score metric
    :param n_jobs: integer, optional. Number of jobs to run in parallel (default 1).
    :param train_sizes: Size of train set
    :param test_score: Classifier score over validation set
    :return: matplotlib object
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score: {0}".format(score))
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=score)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    if test_score is not None:
        plt.plot(train_sizes[-1], test_score, marker='o', color='gray', ls='', label='Test score')

    plt.legend(loc="best")
    return plt


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    """
    Generates a list of windows to search for cars
    :param img: image array
    :param x_start_stop: list with the end and start pixels in the x dimension
    :param y_start_stop: list with the end and start pixels in the y dimension
    :param xy_window: tuple with size of window
    :param xy_overlap: tuple with the overlap windows in both directions
    :return: windows lists
    """
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """
    Draws boxes in an image
    :param img: image array
    :param bboxes: list with the boxes defined as, for example: [((100,100), (200,200)), ((300, 300), (400, 400))]
    :param color: color for the box
    :param thick: box thickness
    :return: image withe the box(es) drawn
    """
    draw_img = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
    return draw_img


def search_windows(img, windows, clf):
    """
    Identify those windows with a Car inside
    :param img: array with image
    :param windows: list with windows defined over image
    :param clf: classifier
    :return: List with windows that have a car
    """

    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        test_features = single_img_features(test_img, spatial_feat=True, hist_feat=True, hog_feat=True)
        # 5) Predict using your classifier
        prediction = clf.predict(test_features.reshape(1, -1))
        # 6) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 7) Return windows for positive detections
    return on_windows


def add_heat(heatmap, bbox_list):
    """
    Create a heatmap with the positives identified
    :param heatmap: heat map
    :param bbox_list: windows with cars inside
    :return:
    """
    # Iterate through list of bboxes
    if bbox_list is None: return heatmap
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    # Return updated heatmap
    return heatmap


def apply_threshold(heatmap, threshold):
    """
    Apply a threshold to the heatmap to set to 0 those values below
    :param heatmap: heat map array
    :param threshold: int threshold value
    :return: heatmap with the threshold applied
    """
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    """
    Draw boxes over labelled boxes
    :param img: array image
    :param labels: labels identifies
    :return: image with boxes included
    """
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


def extract_feats_dataset():
    """
    Process all images in the dataset and generates a pickle object with the associated features and labels arrays.
    Features are distributed in two different folders: vehicles
    :param dataset_path: path to dataset and non-vehicles
    :return: features and dict sizes pickle object
    """
    # Create a list to append feature vectors to
    features = {'vehicles': {}, 'non-vehicles': {}}
    sizes_sources = {'vehicles': {}, 'non-vehicles': {}}
    for label in ['vehicles', 'non-vehicles']:
        for dir, subdir, files in os.walk(DATASET_PATH + label):
            if len(subdir) > 0:
                for source in subdir:
                    features[label][source] = []
            else:
                source = dir[dir.rfind("/") + 1:]
                # features[label][source] = [x for x in files if ".png" in x]
                features[label][source] = [single_img_features(mpimg.imread(os.path.join(dir, x))) for x in files if
                                           ".png" in x]
                sizes_sources[label][source] = len(features[label][source])
                # save results to pandas dataframe:
                pickle.dump([features, sizes_sources], open(DATASET_PATH + "features.p", "wb"))
    return features, sizes_sources
#
#
def create_sets(features, sizes_sources, split=[0.7, 0.2, 0.1]):
    """
    Generate train/test/validation sets
    :param features: list with features
    :param sizes_sources: dict with image sources
    :param split: list with defined set sizes
    :return:
    """
    feat_train, label_train, feat_test, label_test, feat_val, label_val = [], [], [], [], [], []
    for label in ['vehicles', 'non-vehicles']:
        for source in features[label].keys():
            idx_items_train = int(split[0] * sizes_sources[label][source])
            idx_items_test = int(np.sum(split[:2]) * sizes_sources[label][source])
            feat_train.extend(features[label][source][:idx_items_train])
            feat_test.extend(features[label][source][idx_items_train:idx_items_test])
            feat_val.extend(features[label][source][idx_items_test:])
            if label == 'vehicles':
                label_value = 1
            else:
                label_value = 0
            label_train.extend([label_value] * (len(features[label][source][:idx_items_train])))
            label_test.extend([label_value] * (len(features[label][source][idx_items_train:idx_items_test])))
            label_val.extend([label_value] * (len(features[label][source][idx_items_test:])))
    return np.array(feat_train), np.array(label_train), np.array(feat_test), np.array(label_test), np.array(
        feat_val), np.array(label_val)


def run_models(feat_train, label_train, feat_test, label_test):
    """
    Test different model configurations and generates a pickle object with the best performing
    :param feat_train: array with train feats
    :param label_train: array with train labels
    :param feat_test: array with test feats
    :param label_test: array with test labels
    :return: None
    """
    best_f1 = 0
    for num_variables in range(10, 90, 30):
        for C in [1, 10]:
            for tol in [1e-3, 1e-5, 1e-4]:
                pipeline = Pipeline([('scaler', StandardScaler()), ('selection', SelectPercentile(percentile=num_variables)),
                                     ('clf', LinearSVC(random_state=RANDOM_STATE, C=C, tol=tol))])
                pipeline.fit(feat_train, label_train)
                print("Results for Percentile: {0}, C: {1}, tol: {2}:".format(num_variables, C, tol))
                y_pred = pipeline.predict(feat_test)
                print(classification_report(label_test, y_pred, target_names=['non-car', 'car']))
                current_f1 = metrics.f1_score(label_test, y_pred)
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    # save results to pandas dataframe:
                    pickle.dump(pipeline, open(DATASET_PATH + "estimator.p", "wb"))
                    
             
def create_video(path_to_video):
    from moviepy.editor import VideoFileClip
    clip2 = VideoFileClip(path_to_video)
    challenge_clip = clip2.fl_image(process_img)
    challenge_clip.write_videofile(path_to_video.replace(".mp4", "_solved.mp4"), audio=False)
    
    
class BoundingBoxes:
    def __init__(self, n=10):
        # length of queue to store data
        self.n = n
        # hot windows of the last n images
        self.recent_boxes = deque([], maxlen=n)
        # current boxes
        self.current_boxes = None
        self.allboxes = []

    def add_boxes(self):
        self.recent_boxes.appendleft(self.current_boxes)

    def set_current_boxes(self, boxes):
        self.current_boxes = boxes

    def get_all_boxes(self):
        allboxes = []
        for boxes in self.recent_boxes:
            allboxes += boxes
        if len(allboxes) == 0:
            self.allboxes = None
        else:
            self.allboxes = allboxes

    def update(self, boxes):
        self.set_current_boxes(boxes)
        self.add_boxes()
        self.get_all_boxes()

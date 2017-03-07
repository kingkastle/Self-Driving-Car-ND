import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from config import *
import process_data

def plot_signal(img,axis):
    """Simply plot a sample image from dataset"""
    axis.imshow(img)
    axis.set_xticks([])
    axis.set_yticks([])
    return axis


############################################################
# Configuration:
############################################################

seed = 2016
test_size = 0.2

# for reproducibility
np.random.seed(seed)

############################################################
# Load and process Data_test:
############################################################
with open(CURRENT_PATH + 'features_{0}.pickle'.format(VERSION), 'rb') as handle:
    X = pickle.load(handle)
with open(CURRENT_PATH + 'labels_{0}.pickle'.format(VERSION), 'rb') as handle:
    y = pickle.load(handle)

X = X.astype('float32')
y = y.astype('float32')

# Apply augmentation over sequence of data:


# random value:
intensity = np.random.uniform()
intensity = 0.7
# random flipping:
flipping = np.random.choice([True, False])
flipping = True

f, axarr = plt.subplots(2, 2, figsize=(10, 10))
augmented_angles = []
for i in range(0, X.shape[0]*100):
    # random value:
    intensity = np.random.uniform()
    # random flipping:
    flipping = np.random.choice([True, False])
    # random sample
    idx = np.random.randint(X.shape[0])
    _, steering_aug = process_data.augmented_images(X[idx], y[idx], flipping, intensity)
    augmented_angles.append(steering_aug)
    if i in [3000, 8000, 12000, 18000]:
        print('Printing....')
        # the histogram of y labels:
        if i == 3000:
            n, bins, patches = axarr[0, 0].hist(augmented_angles, 100, facecolor='#81F7F3', alpha=0.75)
            axarr[0, 0].set_title('Distribution after {0} iterations'.format(i))
        if i == 8000:
            n, bins, patches = axarr[0, 1].hist(augmented_angles, 100, facecolor='#2E64FE', alpha=0.75)
            axarr[0, 1].set_title('Distribution after {0} iterations'.format(i))
        if i == 12000:
            n, bins, patches = axarr[1, 0].hist(augmented_angles, 100, facecolor='#642EFE', alpha=0.75)
            axarr[1, 0].set_title('Distribution after {0} iterations'.format(i))
        if i == 18000:
            n, bins, patches = axarr[1, 1].hist(augmented_angles, 100, facecolor='#03B3E4', alpha=0.75)
            axarr[1, 1].set_title('Distribution after {0} iterations'.format(i))
            break

f.suptitle('Augmented Data', fontsize=14, fontweight='bold')
f.savefig(PATH_TO_IMG + 'Augmented_distribution.png')


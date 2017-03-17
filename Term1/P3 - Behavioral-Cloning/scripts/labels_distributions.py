import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib import gridspec
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

print(X.shape, y.shape)

# random value:
intensity = np.random.uniform()
intensity = 0.7
# random flipping:
flipping = np.random.choice([True, False])
flipping = True

path_example = '/home/carnd/Self-Driving-Car-ND/Term1/P3 - Behavioral-Cloning/Data/udacity/data/IMG/center_2016_12_01_13_31_15_106.jpg'
img = cv2.imread(path_example, 1)

fig = plt.figure(figsize=(9, 4))
gs = gridspec.GridSpec(1, 2)
ax1 = plt.subplot(gs[0, 0])
ax1.imshow(img)
ax1.set_title('Original Image')
ax1.axis('off')
ax2 = plt.subplot(gs[0, 1])
ax2.imshow(process_data.process_images(path_example, 160, 100, 60))
ax2.axis('off')
ax2.set_title('Resize & Crop')
plt.tight_layout()
fig.savefig(PATH_TO_IMG + 'cropped_image.jpg')

# augmented_images(img, steering_angle, flipping, intensity)
img_proc = process_data.process_images(path_example, 160, 100, 60)
fig = plt.figure(figsize=(14, 5))
gs = gridspec.GridSpec(1, 4)
ax1 = plt.subplot(gs[0, 0])
ax1.imshow(img)
st_angle = -0.4
ax1.set_title('Original\nSteering Angle:{0}'.format(st_angle))
ax1.axis('off')
ax2 = plt.subplot(gs[0, 1])
ax2.imshow(process_data.augmented_images(img_proc, st_angle, True, 0.1)[0])
ax2.axis('off')
ax2.set_title('Augmented example\nSteering Angle:{0}\nFlipping=True\nIntensity=0.1'.format(st_angle*-1),fontsize=11)
ax3 = plt.subplot(gs[0, 2])
ax3.imshow(process_data.augmented_images(img_proc, st_angle, False, 0.5)[0])
ax3.axis('off')
ax3.set_title('Augmented example\nSteering Angle:{0}\nFlipping=False\nIntensity=0.5'.format(st_angle),fontsize=11)
ax4 = plt.subplot(gs[0, 3])
ax4.imshow(process_data.augmented_images(img_proc, st_angle, True, 0.9)[0])
ax4.axis('off')
ax4.set_title('Augmented example\nSteering Angle:{0}\nFlipping=True\nIntensity=0.9'.format(st_angle*-1),fontsize=11)
gs.tight_layout(fig, rect=[0, 0, 1, 1], h_pad=0.5)
fig.savefig(PATH_TO_IMG + 'augmented_example.jpg')



# f, axarr = plt.subplots(2, 2, figsize=(10, 10))
# augmented_angles = []
# for i in range(0, X.shape[0]*100):
#     # random value:
#     intensity = np.random.uniform()
#     # random flipping:
#     flipping = np.random.choice([True, False])
#     # random sample
#     idx = np.random.randint(X.shape[0])
#     _, steering_aug = process_data.augmented_images(X[idx], y[idx], flipping, intensity)
#     augmented_angles.append(steering_aug)
#     if i in [3000, 8000, 12000, 18000]:
#         print('Printing....')
#         # the histogram of y labels:
#         if i == 3000:
#             n, bins, patches = axarr[0, 0].hist(augmented_angles, 100, facecolor='#81F7F3', alpha=0.75)
#             axarr[0, 0].set_title('Distribution after {0} iterations'.format(i))
#         if i == 8000:
#             n, bins, patches = axarr[0, 1].hist(augmented_angles, 100, facecolor='#2E64FE', alpha=0.75)
#             axarr[0, 1].set_title('Distribution after {0} iterations'.format(i))
#         if i == 12000:
#             n, bins, patches = axarr[1, 0].hist(augmented_angles, 100, facecolor='#642EFE', alpha=0.75)
#             axarr[1, 0].set_title('Distribution after {0} iterations'.format(i))
#         if i == 18000:
#             n, bins, patches = axarr[1, 1].hist(augmented_angles, 100, facecolor='#03B3E4', alpha=0.75)
#             axarr[1, 1].set_title('Distribution after {0} iterations'.format(i))
#             break
#
# f.suptitle('Augmented Data', fontsize=14, fontweight='bold')
# f.savefig(PATH_TO_IMG + 'augmented_distribution.png')


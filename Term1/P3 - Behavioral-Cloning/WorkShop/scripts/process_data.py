import pandas
import numpy as np
import cv2
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from scripts.config import *


def brightness_images(image, intensity):
    """
    Function to modify image bright
    :param image: image array
    :param intensity: random value between 0 and 1
    :return: image array modified
    """
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = np.clip(intensity, 0.25, 1.0)
    image1[:, :, 2] = image1[:, :, 2]*random_bright
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1


def random_shadow(image):
    """
    Generate Random Shadows
    :param image: input image
    :return: output image with brightness applied
    """
    top_y = image.shape[1] * np.random.uniform()
    top_x = 0
    bot_x = image.shape[0]
    bot_y = image.shape[1] * np.random.uniform()
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    shadow_mask = 0 * image_hsv[:, :, 2]
    X_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]
    shadow_mask[((X_m - top_x) * (bot_y - top_y) - (bot_x - top_x) * (Y_m - top_y) >= 0)] = 1
    random_bright = .4 + np.random.uniform()
    random_bright = np.clip(random_bright, 0.4, 0.9)  # Between 0.4 and 0.9 is the ideal value to generate shadows
    cond1 = shadow_mask == 1
    cond0 = shadow_mask == 0
    if np.random.randint(2) == 1:
        image_hsv[:, :, 2][cond1] = image_hsv[:, :, 2][cond1] * random_bright
    else:
        image_hsv[:, :, 2][cond0] = image_hsv[:, :, 2][cond0] * random_bright
    image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
    return image


def trans_image(image, steer, trans_range, intensity):
    """
    Function to translate image in both axis
    :param image: image array
    :param steer: steeering angle associated
    :param trans_range: maximum number of pixels to translate
    :param intensity: random value between 0 and 1
    :return: image array modified
    """
    # Translation
    tr_x = trans_range * intensity - trans_range / 2
    steer_ang = steer + tr_x / trans_range * 2 * .2
    tr_y = (trans_range * 0.4) * intensity - (trans_range * 0.4) / 2
    trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    rows, cols = image.shape[:2]
    image_tr = cv2.warpAffine(image, trans_M, (cols, rows))
    return image_tr, steer_ang


def augmented_images(img, steering_angle, flipping, intensity):
    """
    Generates augmented pictures changing brightness, flipping.
    :param img: image array
    :param steering_angle: corresponding steering angle
    :param intensity: numpy random transformation
    :param flipping: Boolean flipping
    :return: augmented image and steering angle
    """
    # brightness image:
    img = brightness_images(img, intensity)
    # random shadows:
    img = random_shadow(img)
    # flip image:
    if flipping:
        img = cv2.flip(img, 1)
        steering_angle *= -1.
    # translate image:
    y_steer = steering_angle
    return img, y_steer


def process_images(path_img, X_PIX, Y_PIX, Y_CROP):
    """
    Resize and Normalize images
    :param path_img: image path
    :param X_PIX: final x pixels size
    :param Y_PIX: final y pixels size
    :param Y_CROP: y pixels to crop image
    :return: processed image
    """
    # read image:
    img = cv2.imread(path_img, 1)
    # normalize image:
    cv2.normalize(img, img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # crop image:
    img = img[Y_CROP:, :, :]
    # resize image:
    img = cv2.resize(img, (X_PIX, Y_PIX), interpolation=cv2.INTER_CUBIC)
    return img


def generate_dataset(threshold_straight=0.0, threshold_straight_near=0.0):
    """
    Process input datasets to generate pickle files to feed neural network
    :param threshold_straight: float number in 0-1 range to determine the portion of straight drives removed
    :param threshold_straight_near: float number in 0-1 range to determine the portion of near straight drives removed
    :return: pickle objects with feature and label sets
    """
    # Read files from multiple sources:
    dataset = []
    for sour in SOURCES:
        print("Loading source: ", sour)
        # load data
        df = pandas.read_csv(SOURCES[sour] + "driving_log.csv", sep=",")

        # some required transformation in the dataset:
        # df["center"] = df['center'].apply(lambda x: SOURCES[sour] + x)
        df[['steering', 'throttle', 'brake', 'speed']] = df[['steering', 'throttle', 'brake', 'speed']].astype(float)

        # Generate a list of tuples where the first element is the numpy array with images and the second element the
        # corresponding steering angle:
        dataset += [(process_images(df.loc[idx, 'center'], X_PIX, Y_PIX, Y_CROP), df.loc[idx, 'steering']) for
                    idx in range(0, df.shape[0])]

    print("Dataset Size: ", len(dataset))

    # Generate X and Y sets for model_CNN training:
    X = np.concatenate([x[0][np.newaxis, :] for x in dataset], axis=0).astype('float32')
    Y = np.asarray([x[1] for x in dataset]).astype('float32')

    # identify straight drives and remove threshold of them:
    straight_drives = np.where(np.abs(Y) < 0.05)[0]
    num_sd = len(straight_drives)
    print('Straight drives removed: ', int(threshold_straight * num_sd))  # 0.7 works ok
    X_bal = np.delete(X, straight_drives[:int(threshold_straight * num_sd)], axis=0)
    Y_bal = np.delete(Y, straight_drives[:int(threshold_straight * num_sd)])

    # identify almost straight drives and remove 60% of them:
    almost_straight_drives = np.where((np.abs(Y) < 0.25) & (np.abs(Y) >= 0.05))[0]
    num_sd = len(almost_straight_drives)
    print('Almost Straight drives removed: ', int(threshold_straight_near * num_sd))  # 0.01 works ok
    X_bal = np.delete(X_bal, almost_straight_drives[:int(threshold_straight_near * num_sd)], axis=0)
    Y_bal = np.delete(Y_bal, almost_straight_drives[:int(threshold_straight_near * num_sd)])

    print("Dataset shape:", X_bal.shape)

    # Visualize labels distributions after artificial balance:
    # generate plot:
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 6), sharey=True)
    f.tight_layout()
    n, bins, patches = ax1.hist(Y, 100, facecolor='#03B3E4', alpha=0.75)
    n, bins, patches = ax2.hist(Y_bal, 100, facecolor='#03B3E4', alpha=0.75)
    ax1.set_title('Original Distribution', fontsize=12)
    ax2.set_title('Balanced Distribution', fontsize=12)
    plt.savefig(PATH_TO_IMG + 'y_distributions.png')

    # convert to float:
    X_bal = X_bal.astype('float32')
    Y_bal = Y_bal.astype('float32')

    # Save files to pickle:
    with open(CURRENT_PATH + 'features_{0}.pickle'.format(VERSION), 'wb') as handle:
        pickle.dump(X_bal, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(CURRENT_PATH + 'labels_{0}.pickle'.format(VERSION), 'wb') as handle:
        pickle.dump(Y_bal, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return


############################################################
# Process Data_test:
############################################################
if __name__ == "__main__":
    print("Processing...")
    generate_dataset(threshold_straight=0.7, threshold_straight_near=0.01)

    print("Process completed!")

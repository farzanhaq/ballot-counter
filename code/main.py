import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import math
import random

from extract_name import get_candidate_name

def convolution(I, filter, mode):
    i_shape = I.shape
    i_rows = i_shape[0]
    i_columns = i_shape[1]

    h_shape = filter.shape
    h_rows = h_shape[0]
    h_columns = h_shape[1]

    f_rows, f_columns = 0, 0

    if mode == 'same':
        f_rows = i_rows
        f_columns = i_columns
    elif mode == 'valid':
        f_rows = i_rows - (h_rows // 2) * 2
        f_columns = i_columns - (h_columns // 2) * 2
    elif mode == 'full':
        f_rows = i_rows + (h_rows // 2) * 2
        f_columns = i_columns + (h_columns // 2) * 2

    filtered_image = np.zeros((f_rows, f_columns))

    for i in range(f_rows):
        image_x_pixel = i  # X coord of the pixel in the input image we center filter at currently
        if mode == 'same':
            image_x_pixel = i
        elif mode == 'valid':
            image_x_pixel = i + (h_rows // 2)
        elif mode == 'full':
            image_x_pixel = i - (h_rows // 2)

        for j in range(f_columns):
            image_y_pixel = j  # Y coord of the pixel in the input image we center filter at
            if mode == 'same':
                image_y_pixel = j
            elif mode == 'valid':
                image_y_pixel = j + (h_columns // 2)
            elif mode == 'full':
                image_y_pixel = j - (h_columns // 2)

            dot_sum = 0

            for k in range(-1 * (h_rows // 2), h_rows // 2 + 1):
                if image_x_pixel + k < 0 or image_x_pixel + k >= i_rows:
                    continue

                for l in range(-1 * (h_columns // 2), h_columns // 2 + 1):
                    if image_y_pixel + l < 0 or image_y_pixel + l >= i_columns:
                        continue
                    filter_val = filter[k + h_rows // 2, l + h_columns // 2]
                    image_val = I[image_x_pixel + k, image_y_pixel + l]
                    dot_sum += filter_val * image_val

            filtered_image[i, j] = dot_sum

    return filtered_image

def segment_ballot(img_path):
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval, img_gray_thresholded = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)

    horizontal_filter = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])

    ballot_segmentation = convolution(img_gray_thresholded, horizontal_filter, 'same')

    ballot_border_top = []
    ballot_border_bottom = []

    for row in range(ballot_segmentation.shape[0]):
        is_border_top = True
        is_border_bottom = True
        for col in range(ballot_segmentation.shape[1]):
            if (ballot_segmentation[row, col] >= 0):
                is_border_top = False
            if (ballot_segmentation[row, col] <= 0):
                is_border_bottom = False
        if (is_border_top and (row-1) not in ballot_border_top):
            ballot_border_top.append(row)
        if (is_border_bottom and (row-1) not in ballot_border_bottom):
            ballot_border_bottom.append(row)

    num_candidates_with_votes = len(ballot_border_top)
    candidate_with_vote = []

    for i in range(num_candidates_with_votes):
        candidate_with_vote.append(img_gray_thresholded[ballot_border_top[i]:ballot_border_bottom[i], 0:img_gray_thresholded.shape[1]])

    vertical_filter = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])
    
    candidate_with_vote_segmentation = convolution(candidate_with_vote[0], vertical_filter, 'same')
    ballot_border_middle = 0

    for col in range(candidate_with_vote_segmentation.shape[1]):
        is_border_middle = True
        for row in range(candidate_with_vote_segmentation.shape[0]):
            if (candidate_with_vote_segmentation[row, col] <= 0):
                is_border_middle = False
        if (is_border_middle and col != 0 and (col - 1) != ballot_border_middle):
            ballot_border_middle = col

    candidate = []
    vote = []

    for i in range(len(candidate_with_vote)):
        candidate.append(candidate_with_vote[i][0:candidate_with_vote[i].shape[0], 0:ballot_border_middle])
        vote.append(candidate_with_vote[i][0:candidate_with_vote[i].shape[0], ballot_border_middle:candidate_with_vote[i].shape[1]])

    return candidate, vote

def augment_data(dir_path, num_images, prefix):
    for j in range(num_images):
        image_path = '{}/{}_votes/{}_vote_{}.png'.format(dir_path, prefix, prefix, j)
        img = cv2.imread(image_path, 0)
        
        img_flip = cv2.flip(img, 1)
        img_noise = img + 2 * img.std() * np.random.random(img.shape)

        gaussian_filter = np.array([
            [1/16, 1/8, 1/16],
            [1/8, 1/4, 1/8],
            [1/16, 1/8, 1/16]
        ])

        img_blur = convolution(img, gaussian_filter, 'same')

        output_flip_path = '{}/{}_votes/{}_vote_flip_{}.png'.format(dir_path, prefix, prefix, j)
        output_noise_path = '{}/{}_votes/{}_vote_noise_{}.png'.format(dir_path, prefix, prefix, j)
        output_blur_path = '{}/{}_votes/{}_vote_blur_{}.png'.format(dir_path, prefix, prefix, j)

        cv2.imwrite(output_flip_path, img_flip)
        cv2.imwrite(output_noise_path, img_noise)
        cv2.imwrite(output_blur_path, img_blur)

def extract_marks(dir_path, num_images, prefix):
    for j in range(num_images):
        image_path = '{}/{}/{}_{}.png'.format(dir_path, prefix, prefix, j)
        candidate, vote = segment_ballot(image_path)

        output_path = '{}/{}_votes/{}_vote_{}.png'.format(dir_path, prefix, prefix, j)
        cv2.imwrite(output_path, vote[0])

"""def crop_images():
    images = [cv2.imread(file) for file in glob.glob("dataset/valid/*.png")]
    for image in range(len(images)):
        cv2.imwrite('dataset/valid2/valid_' + str(image) + '.png', images[image][2:-2, 4:-4])

    images = [cv2.imread(file) for file in glob.glob("dataset/invalid/*.png")]
    for image in range(len(images)):
        cv2.imwrite('dataset/invalid2/invalid_' + str(image) + '.png', images[image][2:-2, 4:-4])"""
        
#crop_images()
#extract_marks('../dataset', 150, 'invalid')
#extract_marks('../dataset', 50, 'valid')
#augment_data('../dataset', 150, 'invalid')
#augment_data('../dataset', 50, 'valid')
#augment_single_data()
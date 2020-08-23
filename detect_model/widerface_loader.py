from utils import drawplt
import tensorflow as tf
import numpy as np
import sys
import os


def process_image(image_path, target_w, target_h):
    """
    load, resize, normalize image to [-1, 1]
    """
    image = tf.keras.preprocessing.image.load_img(
        image_path, color_mode='rgb', target_size=(target_w, target_h),
        interpolation='bilinear'
    )
    image = np.array(image)
    image_normalized = image / 127.5 - 1.0
    return image_normalized


def print_progress(num):
    sys.stdout.write('\rProcessing: ' + str(num))
    sys.stdout.flush()
    return num + 1


def load_widerface(gt_dir, train_dir, target_w=128, target_h=128,
                   min_face_size=7, filter_entire_img=True):
    """
    loads widerface dataset from directory. filter out images with small faces.
    returns: [num_picture, width(128), height(128), 3], [num_picture, num_gt, 4] 
    (cx, cy, w, h - gt pixel location of resized image)
    """
    images, labels = [], []
    print('Processing dataset...')
    with open(gt_dir, 'r') as f:
        process_num = 1
        while True:
            process_num = print_progress(process_num)
            image_name = f.readline().strip("\n ")

            # break if end of line
            if not image_name:
                break

            num_bbox = int(f.readline())

            # continue if no gt exists in image
            if num_bbox == 0:
                f.readline()
                continue

            image_path = os.path.join(train_dir, image_name)
            image = np.array(tf.keras.preprocessing.image.load_img(image_path))
            scale_x = image.shape[1] / target_w
            scale_y = image.shape[0] / target_h

            # scale gt
            label = []
            filter_flag = False
            for bbox in range(num_bbox):
                gt_str = f.readline().strip('\n ').split(' ')
                gt = [int(i) for i in gt_str]
                gt[0], gt[2] = gt[0] / scale_x,  gt[2] / scale_x
                gt[1], gt[3] = gt[1] / scale_y,  gt[3] / scale_y

                # filter out invalid or small gt boxes
                if min(gt[2], gt[3]) > min_face_size and gt[7] != 1:
                    label.append(np.array(gt[:4]))
                else:
                    filter_flag = True

            if filter_flag and filter_entire_img:
                continue

            image_normalized = process_image(
                image_path, target_w, target_h)

            label = np.array(label)
            if label.size > 0:
                images.append(image_normalized)
                labels.append(label)
                # drawplt(image_normalized, label)

        images = np.array(images)
        labels = np.array(labels)
        print('\nLoaded: ', images.shape, labels.shape)

        return images, labels

from utils.utils import drawplt, calc_iou_batch
import tensorflow as tf
import numpy as np
import sys
import os


def read_image(image_path, target_w, target_h):
    image = tf.keras.preprocessing.image.load_img(
        image_path, color_mode='rgb', target_size=(target_w, target_h),
        interpolation='bilinear'
    )
    return np.array(image)


def process_image(image_path, target_w, target_h):
    """
    load, resize, normalize image to [-1, 1]
    """
    image = read_image(image_path, target_w, target_h)
    image_normalized = image / 127.5 - 1.0
    return image_normalized


def print_progress(num):
    sys.stdout.write('\rProcessing: ' + str(num))
    sys.stdout.flush()
    return num + 1


def load_widerface(gt_dir, train_dir, target_w=128, target_h=128,
                   min_face_size=7, filter_entire_img=True):
    """
    loads widerface dataset from directory. filter out images with small faces.\n
    returns: [num_picture, width(128), height(128), 3], [num_picture, num_gt, 4]\n
    4(cx, cy, w, h - ratio of pixel location relative to resized image)
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
            scale_x = image.shape[1]
            scale_y = image.shape[0]

            # scale gt
            label = []
            filter_flag = False
            for bbox in range(num_bbox):
                gt_str = f.readline().strip('\n ').split(' ')
                gt = [int(i) for i in gt_str]
                gt[0], gt[2] = gt[0] / scale_x,  gt[2] / scale_x
                gt[1], gt[3] = gt[1] / scale_y,  gt[3] / scale_y

                # filter out invalid or small gt boxes
                if (min(gt[2] * target_w, gt[3] * target_h) > min_face_size and
                        gt[7] != 1):
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
                # show normalized image and bbox
                # drawplt(image_normalized, label, target_w, target_h)

        images = np.array(images)
        labels = np.array(labels)
        print('\nLoaded: ', images.shape, labels.shape)

        return images, labels


def generate_gt(labels, anchors, iou_threshold=0.5):
    """
    labels: [num_labels, num_gt, 4]]\n
    anchors: [num_box, 4(cx, cy, w, h)]\n
    returns: [num_labels, num_boxes, 5(responsible, tcx, tcy, tw, th)]
    """
    num_boxes = anchors.shape[0]
    process_num = 0
    gts = np.empty([0, num_boxes, 5])

    print('Processing gt...')
    for label in labels:
        process_num = print_progress(process_num)
        gt = np.zeros([num_boxes, 5], dtype=np.float32)
        for box in label:
            ious = np.array(calc_iou_batch(box, anchors))
            # Match higher than threshold
            maxarg = ious > iou_threshold
            # Match best jaccard(IOU) overlap
            maxarg[tf.argmax(ious)] = True
            # translate to target
            gt[maxarg, 0] = 1.0
            gt[maxarg, 1:3] = (box[0:2] - anchors[maxarg, 0:2]
                               ) / anchors[maxarg, 0:2]
            gt[maxarg, 3:5] = np.log(box[2:4] / anchors[maxarg, 2:4])
        gts = np.vstack([gts, np.expand_dims(gt, axis=0)])
    print('\nLoaded: ', gts.shape)
    return gts


def dataloader(images, ground_truths, batch_size=64):
    """
    images: [num_images, 128, 128, 3]\n
    ground_truths: [num_labels, num_boxes, 5(conf, tcx, tcy, tw, th)]\n
    returns: ([batch_size, 128, 128, 3], [batch_size, num_boxes, 5])
    """
    data_keys = np.arange(len(images))
    while True:
        selected_keys = np.random.choice(
            data_keys, replace=False, size=batch_size)

        image_batch = []
        gt_batch = []
        for key in selected_keys:
            image_batch.append(images[key])
            gt_batch.append(ground_truths[key])
        yield (np.array(image_batch, dtype=np.float32),
               np.array(gt_batch, dtype=np.float32))

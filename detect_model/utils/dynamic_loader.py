from utils import calc_iou_batch, drawplt, prediction_to_bbox, tie_resolution
from PIL import Image
import tensorflow as tf
import numpy as np
import sys
import os


def read_image(image_path, target_w, target_h):
    # tf.keras.preprocessing.image.smart_resize()
    image = tf.keras.preprocessing.image.load_img(
        image_path, color_mode='rgb', target_size=[target_w, target_h])
    return np.array(image)


def process_image_batch(image, target_w, target_h):
    """
    load, resize, normalize image to [-1, 1]
    """
    image = tf.image.resize(image, [target_w, target_h])
    image_normalized = image / 127.5 - 1.0
    return image_normalized


def print_progress(num):
    sys.stdout.write('\rProcessing: ' + str(num))
    sys.stdout.flush()
    return num + 1


def load_widerface(gt_dir, train_dir, min_face_ratio=0.01, filter_entire_img=True):
    """
    loads widerface dataset from directory. filter out images with small faces.\n
    returns: [num_picture] image_dirs, [num_picture, num_gt, 4]\n
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
            image = Image.open(image_path)
            image_w, image_h = image.size

            # scale gt
            label = []
            filter_flag = False
            for bbox in range(num_bbox):
                gt_str = f.readline().strip('\n ').split(' ')
                gt = [int(i) for i in gt_str]
                gt[0], gt[2] = (gt[0] + gt[2] / 2) / image_w,  gt[2] / image_w
                gt[1], gt[3] = (gt[1] + gt[3] / 2) / image_h,  gt[3] / image_h

                # filter out invalid or small gt boxes
                if (gt[2] * gt[3] > min_face_ratio and gt[4] != 2
                        and gt[7] != 1 and gt[8] != 2):
                    label.append(np.array(gt[:4]))
                else:
                    filter_flag = True

            if filter_flag and filter_entire_img:
                continue

            label = np.array(label)
            if label.size > 0:
                images.append(image_path)
                labels.append(label)

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

    for label in labels:
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
                               ) / anchors[maxarg, 2:4]
            gt[maxarg, 3:5] = np.log(box[2:4] / anchors[maxarg, 2:4])
        gts = np.vstack([gts, np.expand_dims(gt, axis=0)])
    return gts


def dynamic_dataloader(image_urls, labels, anchors, target_w, target_h, batch_size=64):
    """
    image_urls: [num_images] contains image url\n
    labels: [num_labels, num_gt, 4]]\n
    returns: ([batch_size, 128, 128, 3], [batch_size, num_boxes, 5])\n
    this function dynamically loads image from image_url, and make gt from labels.
    """
    data_keys = np.arange(len(image_urls))
    while True:
        selected_keys = np.random.choice(
            data_keys, replace=False, size=batch_size)

        image_batch = []
        label_batch = []
        for key in selected_keys:
            image = read_image(image_urls[key], target_w, target_h)
            image_batch.append(image)
            label_batch.append(labels[key])

        # TODO: do augmentation

        gt_batch = generate_gt(label_batch, anchors)
        image_batch = process_image_batch(
            image_batch, target_w, target_h)

        yield (np.array(image_batch, dtype=np.float32),
               np.array(gt_batch, dtype=np.float32))


# test code
if __name__ == "__main__":
    anchors = np.load(os.path.join("./", "anchors.npy"))

    model = tf.keras.models.load_model(
        './Blazeface158.hdf5', compile=False)

    i, l = load_widerface("/home/cyrojyro/hddrive/wider_face_split/wider_face_train_bbx_gt.txt",
                          "/home/cyrojyro/hddrive/WIDER_train/images")
    anchors = np.load(os.path.join("./", "anchors.npy"))
    dl = dynamic_dataloader(i, l, anchors, 128, 128)
    for _ in range(10):
        a, gt = next(dl)  # a= 64이미지 gt= 64개 anchor 상대적 거리 gt
        p = prediction_to_bbox(gt, anchors)  # p = 64개 이미지 gt
        for j in range(64):
            ress = p[j]
            ress = ress[ress[..., 0] == 1]
            drawplt(a[j], ress[..., 1:5], 128, 128)

            prediction = model(np.expand_dims(a[j], 0), training=False)
            prediction = np.array(prediction)
            bbox = prediction_to_bbox(prediction, anchors)[0]
            bbox = bbox[bbox[..., 0] > 0.5]
            resolved_boxes = tie_resolution(
                bbox, 0.5, 0.1)
            drawplt(a[j], resolved_boxes, 128, 128)

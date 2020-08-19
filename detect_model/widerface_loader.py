from utils import drawplt
import tensorflow as tf
import numpy as np
import os


WIDER_GT = "/home/cyrojyro/hddrive/wider_face_split/wider_face_train_bbx_gt.txt"
WIDER_TRAIN = "/home/cyrojyro/hddrive/WIDER_train/images"


def load_widerface(target_w=128, target_h=128, min_face_size=0, filter_entire_img=True):
    images = []
    labels = []
    with open(WIDER_GT, 'r') as f:
        while True:
            image = f.readline().strip("\n ")
            if not image:
                break
            print(image)
            num_bbox = int(f.readline())
            image_path = os.path.join(WIDER_TRAIN, image)
            image = np.array(tf.keras.preprocessing.image.load_img(image_path))

            scale_x = image.shape[1] / target_w
            scale_y = image.shape[0] / target_h

            label = []

            # if no gt exists in image
            if num_bbox == 0:
                f.readline()
                continue

            # scale gt
            filter_flag = False
            for bbox in range(num_bbox):
                gt = f.readline().strip('\n ')
                splitted_gt = gt.split(' ')
                gt = [int(i) for i in splitted_gt]
                gt[0], gt[2] = gt[0] / scale_x,  gt[2] / scale_x
                gt[1], gt[3] = gt[1] / scale_y,  gt[3] / scale_y
                # filter invalid or small gt boxes
                if gt[2] > min_face_size and gt[3] > min_face_size and gt[7] != 1:
                    label.append(np.array(gt[:4]))
                else:
                    filter_flag = True

            if filter_flag and filter_entire_img:
                continue

            # scaling, preprocessing[-1, 1]
            image = tf.keras.preprocessing.image.load_img(
                image_path, color_mode='rgb', target_size=(target_w, target_h),
                interpolation='bilinear'
            )
            image = np.array(image)
            image_normalized = image / 127.5 - 1.0

            label = np.array(label)
            if label.size > 0:
                images.append(image_normalized)
                labels.append(label)
                # drawplt(image, label)

        images = np.array(images)
        labels = np.array(labels)
        print(images.shape, labels.shape, sep="\n")

        return images, labels


if __name__ == "__main__":
    load_widerface(min_face_size=7)

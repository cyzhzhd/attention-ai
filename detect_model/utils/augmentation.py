import tensorflow as tf
import random


def random_flip(image, labels, prob=0.5):
    if random.random() > prob:
        image = tf.image.flip_left_right(image)
        for label in labels:
            label[0] = 1.0 - label[0]
    return image, labels


def random_brightness(image, max_factor=0.25, prob=0.5):
    if random.random() > prob:
        factor = (random.random() - 0.5) * 2 * max_factor
        image = tf.image.adjust_brightness(image, factor)
    return image

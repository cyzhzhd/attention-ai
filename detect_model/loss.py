import tensorflow as tf


class MultiboxLoss(tf.keras.losses.Loss):
    def __init__(self):
        pass

    def __call__(self, a, b):
        print(a, b)

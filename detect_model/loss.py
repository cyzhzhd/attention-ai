import tensorflow as tf


def MultiboxLoss(true, pred):
    tf.print(tf.keras.backend.shape(pred))
    tf.print(tf.keras.backend.shape(true))
    return tf.keras.backend.mean(tf.keras.backend.pow(pred - true, 2))

import tensorflow as tf


class ChannelPad(tf.keras.layers.Layer):
    def __init__(self):
        super(ChannelPad, self).__init__()

    def call(self, inputs, pad_num):
        assert pad_num >= 0
        if pad_num == 0:
            return inputs
        pad_shape = tf.concat([inputs.shape[:-1], [pad_num]], axis=-1)
        padding = tf.zeros(pad_shape)
        return tf.concat([inputs, padding], axis=-1)


class BlazeBlock(tf.keras.Model):
    def __init__(self, filters_1, filters_2=None, kernel_size=5,
                 strides=1, padding='same', activation=tf.nn.relu):
        super(BlazeBlock, self).__init__()
        assert strides in [1, 2]
        self.use_residual = (strides == 2)
        self.use_double_block = filters_2 is not None
        self.activation = tf.keras.layers.Activation(activation)

        self.dw_conv_1 = tf.keras.Sequential([
            tf.keras.layers.SeparableConv2D(
                filters=filters_1,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
            ),
            tf.keras.layers.BatchNormalization()
        ])
        if self.use_double_block:
            self.dw_conv_2 = tf.keras.Sequential([
                tf.keras.layers.SeparableConv2D(
                    filters=filters_2,
                    kernel_size=kernel_size,
                    strides=1,
                    padding=padding,
                ),
                tf.keras.layers.BatchNormalization()
            ])

    def call(self, inputs, training=False):
        x_0 = self.dw_conv_1(inputs)
        if self.use_double_block:
            x_0 = self.activation(x_0)
            x_0 = self.dw_conv_2(x_0)
        if self.use_residual:
            pad_num = x_0.shape[-1] - inputs.shape[-1]
            x_1 = tf.keras.layers.MaxPool2D()(inputs)
            x_1 = ChannelPad()(x_1, pad_num)
            x_0 = tf.keras.layers.Add()([x_0, x_1])
        return self.activation(x_0)


class Backbone(tf.keras.Model):
    def __init__(self, activation=tf.nn.relu):
        super(Backbone, self).__init__()
        self.activation = activation

    def call(self, inputs):
        x_0 = tf.keras.layers.Conv2D(
            filters=24, kernel_size=5, strides=2, padding='same')(inputs)
        x_0 = tf.keras.layers.BatchNormalization()(x_0)
        x_0 = tf.keras.layers.Activation(self.activation)(x_0)

        x_0 = BlazeBlock(filters_1=24)(x_0)
        x_0 = BlazeBlock(filters_1=24)(x_0)
        x_0 = BlazeBlock(filters_1=24, strides=2)(x_0)
        x_0 = BlazeBlock(filters_1=48)(x_0)
        x_0 = BlazeBlock(filters_1=48)(x_0)

        x_0 = BlazeBlock(filters_1=24, filters_2=96, strides=2)(x_0)
        x_0 = BlazeBlock(filters_1=24, filters_2=96)(x_0)
        x_0 = BlazeBlock(filters_1=24, filters_2=96)(x_0)
        x_1 = BlazeBlock(filters_1=24, filters_2=96, strides=2)(x_0)
        x_1 = BlazeBlock(filters_1=24, filters_2=96)(x_1)
        x_1 = BlazeBlock(filters_1=24, filters_2=96)(x_1)

        return x_0, x_1


class BlazeFace(tf.keras.Model):
    def __init__(self, activation=tf.nn.relu):
        super(BlazeFace, self).__init__()
        self.activation = activation
        self.boxes_num = [2, 6]
        self.cell_size = [16, 8]

    def call(self, inputs):
        feat_map_0, feat_map_1 = Backbone(activation=self.activation)(inputs)

        x_0 = tf.keras.layers.Conv2D(filters=self.boxes_num[0], kernel_size=3,
                                     padding='same')(feat_map_0)
        x_0 = tf.keras.layers.Reshape(
            [self.cell_size[0]**2 * self.boxes_num[0], 1])(x_0)

        x_1 = tf.keras.layers.Conv2D(filters=self.boxes_num[1], kernel_size=3,
                                     padding='same')(feat_map_1)
        x_1 = tf.keras.layers.Reshape(
            [self.cell_size[1]**2 * self.boxes_num[1], 1])(x_1)
        confidences = tf.concat([x_0, x_1], axis=1)

        x_2 = tf.keras.layers.Conv2D(filters=self.boxes_num[0] * 4, kernel_size=3,
                                     padding='same')(feat_map_0)
        x_2 = tf.keras.layers.Reshape(
            [self.cell_size[0]**2 * self.boxes_num[0], 4])(x_2)

        x_3 = tf.keras.layers.Conv2D(filters=self.boxes_num[1] * 4, kernel_size=3,
                                     padding='same')(feat_map_1)
        x_3 = tf.keras.layers.Reshape(
            [self.cell_size[1]**2 * self.boxes_num[1], 4])(x_3)
        bboxes = tf.concat([x_2, x_3], axis=1)

        return tf.concat([confidences, bboxes], axis=-1)


py

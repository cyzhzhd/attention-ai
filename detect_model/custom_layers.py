import tensorflow as tf


class ChannelPad(tf.keras.layers.Layer):
    def __init__(self):
        super(ChannelPad, self).__init__()

    def call(self, inputs, pad_num):
        assert pad_num >= 0
        if pad_num == 0:
            return inputs
        padding = tf.zeros_like(inputs)
        padding = padding[..., :pad_num]
        #pad_shape = tf.concat([inputs.shape[:-1], [pad_num]], axis=-1)
        #padding = tf.zeros(pad_shape)
        return tf.concat([inputs, padding], axis=-1)


class BlazeBlock(tf.keras.layers.Layer):
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

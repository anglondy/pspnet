import tensorflow as tf
from proj_pspnet.layers import InterpolationLayer


class InterpolateStep(tf.keras.layers.Layer):
    def __init__(self, size, k):
        super().__init__()
        self._size = size
        if k == 10:
            self._output_shape = (10, 10)
            self._kernel = {
                1: 10,
                2: 5,
                3: 4,
                6: 5
            }
            self._strides = {
                1: 1,
                2: 5,
                3: 3,
                6: 1
            }

        if k == 38:
            self._output_shape = (38, 38)
            self._kernel = {
                1: 38,
                2: 19,
                3: 13,
                6: 7
            }
            self._strides = {
                1: 1,
                2: 19,
                3: 12,
                6: 6
            }

        self.avg_pool = tf.keras.layers.AveragePooling2D(self._kernel[self._size], self._strides[self._size])

        self.conv1 = tf.keras.layers.Conv2D(512, (1, 1), strides=(1, 1), use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.Activation('relu')

        self.inter = InterpolationLayer(self._output_shape)

    def call(self, inputs):
        inputs = self.avg_pool(inputs)

        inputs = self.conv1(inputs)
        inputs = self.bn(inputs)
        inputs = self.activation(inputs)

        return self.inter(inputs)

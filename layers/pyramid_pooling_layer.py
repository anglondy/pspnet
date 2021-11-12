import tensorflow as tf
from proj_pspnet.layers import InterpolationLayer, InterpolateStep


class PyramidPoolModel(tf.keras.layers.Layer):
    def __init__(self, k):
        super().__init__()
        self.k = k

        self.inter_1 = InterpolateStep(1, self.k)
        self.inter_2 = InterpolateStep(2, self.k)
        self.inter_3 = InterpolateStep(3, self.k)
        self.inter_6 = InterpolateStep(6, self.k)

        self.concat = tf.keras.layers.Concatenate(axis=-1)

    def call(self, inputs):
        inter_1 = self.inter_1(inputs)
        inter_2 = self.inter_2(inputs)
        inter_3 = self.inter_3(inputs)
        inter_6 = self.inter_6(inputs)

        concat = self.concat([inputs, inter_1, inter_2, inter_3, inter_6])
        return concat

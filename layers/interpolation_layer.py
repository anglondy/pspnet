import tensorflow as tf


class InterpolationLayer(tf.keras.layers.Layer):
    def __init__(self, output_shape):
        super().__init__()
        self._output_shape = tf.Variable(initial_value=output_shape, trainable=False)

    def call(self, inputs):
        inputs = tf.image.resize(inputs, size=self._output_shape)
        return inputs

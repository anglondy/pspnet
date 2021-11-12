import tensorflow as tf
from proj_pspnet.layers import *


class PSPNet(tf.keras.Model):
    def __init__(self, input_shape: tuple = (300, 300, 3), num_classes: int = 151, train_resnet: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.train_resnet = train_resnet
        self._input_shape = input_shape

        resnet = tf.keras.applications.ResNet50(include_top=False, input_shape=self._input_shape, weights='imagenet')
        self.resnet = tf.keras.Model(inputs=resnet.input, outputs=resnet.get_layer('conv3_block4_out').output)

        if not self.train_resnet:
            self.resnet.trainable = False

        self.pyramid_pool_model = PyramidPoolModel(k=38)
        self.conv1 = tf.keras.layers.Conv2D(512, 3, padding="same", use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization()
        self.activation1 = tf.keras.layers.Activation('relu')
        self.drop = tf.keras.layers.Dropout(0.1)

        self.conv2 = tf.keras.layers.Conv2D(self.num_classes, 1)

        self.interpol = InterpolationLayer([self._input_shape[0], self._input_shape[1]])
        self.activation2 = tf.keras.layers.Activation('softmax')

    def call(self, inputs):
        x = self.resnet(inputs)
        x = self.pyramid_pool_model(x)

        x = self.conv1(x)
        x = self.bn(x)
        x = self.activation1(x)

        x = self.drop(x)

        x = self.conv2(x)
        x = self.interpol(x)

        return self.activation2(x)

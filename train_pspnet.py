import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from proj_pspnet.data import DataPrep
from proj_pspnet.constants.constants import IMAGES_PATH, LABELS_PATH, INPUT_SHAPE, NOT_RARE_CLASSES
from proj_pspnet.pspnet import PSPNet

if __name__ == '__main__':
    # preparing data
    x_images = DataPrep.get_all_data(IMAGES_PATH, IMAGES_PATH, target_shape=(300, 300), data_type=np.float16)
    y_images = DataPrep.get_all_data(IMAGES_PATH, LABELS_PATH, target_shape=(300, 300), data_type=np.float16,
                                     method=True, standardize=False)

    # label shape should be (None, 300, 300, 1)
    y_images = y_images[..., 0:1]

    # cleaning from rare classes
    x_images, y_images = DataPrep.clean_from_rare_classes(x_images, y_images, np.array(NOT_RARE_CLASSES))

    # splitting data vor training and evaluating
    x_train, x_test, y_train, y_test = train_test_split(x_images, y_images, test_size=0.1, random_state=42)

    net = PSPNet(input_shape=INPUT_SHAPE)
    net.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=3e-3),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    num_examples_train = 500

    epochs = 2
    semi_epochs = 3

    num_epochs = int((x_train.shape[0] - 1) // num_examples_train + 1)
    num_examples_test = int((x_test.shape[0] - 1) // num_epochs + 1)

    for _ in range(epochs):
        for i in range(num_epochs):
            net.fit(
                x_train[num_examples_train * i:num_examples_train * (i + 1)].astype(np.float32),
                y_train[num_examples_train * i:num_examples_train * (i + 1)].astype(np.float32),
                validation_data=(
                    x_test[num_examples_test * i: num_examples_test * (i + 1)],
                    y_test[num_examples_test * i: num_examples_test * (i + 1)]),
                epochs=semi_epochs,
                batch_size=10,
                validation_batch_size=10
                )

    net.save('models/PSPNet_v4.tf', save_format='tf')

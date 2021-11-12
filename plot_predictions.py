import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from proj_pspnet.data import DataPrep
from proj_pspnet.layers import *
from proj_pspnet.constants.constants import *


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

    net = tf.keras.models.load_model(
        '/content/drive/MyDrive/PSPNet_v4.tf',
        custom_objects={
            'PyramidPoolModel': PyramidPoolModel,
            'InterpolateStep': InterpolateStep,
            'InterpolationLayer': InterpolationLayer
        })

    start_pos, end_pos = 0, 10
    predictions = DataPrep.decode_predictions(net.predict(x_test[start_pos:end_pos])).astype(np.float32)
    true_mask = np.where(y_test[start_pos:end_pos] == predictions, 1.0, 0.0)

    for i in range(10):
        figure = plt.figure(figsize=(10, 5))
        ax1 = figure.add_subplot(121)
        ax2 = figure.add_subplot(122)
        print('Accuracy =', true_mask[i].mean())
        ax1.imshow(x_test[i].astype(np.float32))
        ax2.imshow(true_mask[i])
        plt.show()

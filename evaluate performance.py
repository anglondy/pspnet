import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from proj_pspnet.data import DataPrep
from proj_pspnet.constants.constants import IMAGES_PATH, LABELS_PATH, NOT_RARE_CLASSES
from proj_pspnet.layers import *
from proj_pspnet.utils.model_performance import evaluate_performance


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

    tr_acc, tr_cls, tr_apc, tr_tot_ppc, tr_ppg = evaluate_performance(net, x_train, y_train)
    print('Overall pixel-wise accuracy (train set) =', tr_acc)
    for i in range(len(tr_cls)):
        print(f'''for class {tr_cls[i]}: 
            accuracy = {tr_apc[i]:03f}%, 
            contains {tr_tot_ppc[i]:03f}% of all pixels in train set, 
            can gain up to {tr_ppg[i]:03f}%''')

    tst_acc, tst_cls, tst_apc, tst_tot_ppc, tst_ppg = evaluate_performance(net, x_test, y_test)
    print('Overall pixel-wise accuracy (test set) =', tst_acc)
    for i in range(len(tst_cls)):
        print(f'''for class {tst_cls[i]}: 
            accuracy = {tst_apc[i]:03f}%, 
            contains {tst_tot_ppc[i]:03f}% of all pixels in test set, 
            can gain up to {tst_ppg[i]:03f}%''')

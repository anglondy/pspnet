import tensorflow as tf
import os
import numpy as np


class DataPrep:
    @staticmethod
    def get_image(path: str, data_type, standardize: bool) -> np.array:
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3).numpy().astype(data_type)

        if standardize:
            return image / 255

        return image

    @staticmethod
    def get_all_data(order_path: str, path_info: str, target_shape: tuple = (300, 300), data_type: type = np.float32,
                     method: bool = False, standardize: bool = True) -> np.array:
        extension = os.listdir(path_info)[0][-3:]
        x = []

        for i in os.listdir(order_path):
            path = os.path.join(path_info, i[:-3]) + extension
            image = DataPrep.get_image(path, data_type=data_type, standardize=standardize)

            if not method:
                x.append(tf.image.resize(image, size=target_shape).numpy())
            else:
                x.append(tf.image.resize(image, size=target_shape, method='nearest').numpy())

        return np.array(x, dtype=data_type)

    @staticmethod
    def clean_from_rare_classes(x_set: np.array, y_set: np.array, mask: np.array) -> (np.array, np.array):
        lst = [[] for _ in range(151)]

        for num in range(y_set.shape[0]):
            for i in range(151):
                if [i] in y_set[num]:
                    lst[i].append(num)

        images = []
        for num, curr_mask in enumerate(mask):
            if curr_mask:
                continue

            for i in lst[num]:
                if i not in images:
                    images.append(i)

        true_mask = [i for i in range(y_set.shape[0]) if i not in images]

        return x_set[true_mask], y_set[true_mask]

    @staticmethod
    def augment_data(image_data_gen, _x_batch: np.array, _y_batch: np.array,
                     shape=(300, 300, 3)) -> (np.array, np.array):

        _x_batch, _y_batch = _x_batch.copy(), _y_batch.copy()
        _y_batch = np.concatenate((_y_batch, _y_batch, _y_batch), axis=-1)

        for i in range(_x_batch.shape[0]):
            transform = image_data_gen.get_random_transform(shape)
            _x_batch[i] = image_data_gen.apply_transform(_x_batch[i], transform)
            _y_batch[i] = image_data_gen.apply_transform(_y_batch[i], transform)

        return _x_batch, _y_batch[..., 0:1]

    @staticmethod
    def unison_shuffled_copies(a, b):
        p = np.random.permutation(len(a))
        return a.copy()[p], b.copy()[p]

    @staticmethod
    def decode_predictions(y_batch: np.array):
        y_batch = np.argmax(y_batch, axis=-1).reshape(-1, 300, 300, 1)
        return y_batch

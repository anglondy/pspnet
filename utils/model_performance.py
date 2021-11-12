import numpy as np
from proj_pspnet.data import DataPrep
from proj_pspnet.constants.constants import NOT_RARE_CLASSES


def evaluate_performance(model, x_set: np.array, y_set: np.array):
    predictions = DataPrep.decode_predictions(model.predict(x_set)).astype(np.float16)
    mask = np.where(predictions == y_set, False, True)
    errors = y_set[mask]

    unique_true, counts_true = np.unique(y_set, return_counts=True)
    unique_pred, counts_pred = np.unique(errors, return_counts=True)

    total_pix = x_set.shape[0] * x_set.shape[1] * x_set.shape[2]
    accuracy = 1 - errors.shape[0] / total_pix

    classes, accuracy_per_class, total_pix_per_class, possible_percentage_gain = [], [], [], []

    for i in range(len(counts_pred)):
        if NOT_RARE_CLASSES[i]:
            classes.append(unique_pred[i])
            accuracy_per_class.append(counts_pred[i] / counts_true[i] * 100)
            total_pix_per_class.append(counts_true[i] / total_pix * 100)
            possible_percentage_gain.append(((counts_true[i] - counts_pred[i]) / total_pix) * 100)

    return accuracy, classes, accuracy_per_class, total_pix_per_class, possible_percentage_gain

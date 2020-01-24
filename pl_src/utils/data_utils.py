# -*- coding: utf-8 -*-
"""
@created on: 1/23/20,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""

import pandas as pd
from pl_src.utils.constants import StringConstants, ModelConstants
import numpy as np


def read_csv(path):
    """
    Read csv file
    :param path: Absolute filepath of data
    :return: data
    """
    if path is None:
        raise Exception(StringConstants.FILEPATH_EMPTY)
    return pd.read_csv(path)


def data_prep(data, train=True):
    """
    Split data into input and labels. This method assumes label is one column and is at the end of file
    :param data: Combined data (Usually data fresh ut of a file)
    :param train: Is this method used in training/testing(True) or serving(False).
                If it is used on training/testing if assumes last column as label.
                If it is used for serving, in which case the value should be false, the method assumes data has no label column.
    :return: Two lists. One containing input and other containing labels
    """
    if data is None or len(data) == 0:
        raise Exception(StringConstants.DATA_READ_FAILED)

    data = np.array(data)
    if train:
        features_len = data.shape[
                           1] - 1  # This -1 is because last column is label and we just need len of input features.
        inp_data, labels = data[:, :features_len], data[:, -1]
        return inp_data, labels
    else:
        data = np.reshape(data, (-1, 4))
        return data


def convert_label_to_onehot(labels):
    """
    Converts a given set of strings to their onehot representation
    :param labels: String labels which needs to be converted to onehot vectors based on some configuration. For now, the configuration will be hardcoded in this method.
    :return: Onehot vectors
    """

    def get_onehot_for_index(n, index):
        onehot = np.zeros(n)
        onehot[index] = 1
        return onehot

    # This is a completely noob idea. But for time being, it will work. Voila !!
    config = ModelConstants.ONEHOT_CONFIG
    unique_labels = len(config)
    return [get_onehot_for_index(unique_labels, config.index(x)) for x in labels]


def convert_onehot_to_label(onehot):
    """
    Converts onehot vector back to its original labels
    :param onehot: Onehot vectors
    :return: Original string labels
    """
    if onehot is None or len(onehot) == 0:
        raise Exception(StringConstants.DATA_HAS_NO_CONTENT)
    config = ModelConstants.ONEHOT_CONFIG
    if len(onehot) == 1:
        return config[np.where(onehot[0] == 1)[0][0]]
    else:
        indices = np.where(onehot == 1)[1]
        return tuple(config[x] for x in indices)

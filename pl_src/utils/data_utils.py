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
from pl_src.utils.string_constants import StringConstants
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


def data_prep_for_training(data):
    """
    Split data into input and labels. This method assumes label is one column and is at the end of file
    :param data: Combined data (Usually data fresh ut of a file)
    :return: Two lists. One containing input and other containing labels
    """
    if data is None or len(data) == 0:
        raise Exception(StringConstants.DATA_READ_FAILED)
    features_len = data.shape[
                       1] - 1  # This -1 is because last column is label and we just need len of input features.
    data = np.array(data)
    inp_data, labels = data[:, :features_len], data[:, -1]
    return inp_data, labels


def convert_to_onehot(labels):
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
    config = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    unique_labels = len(config)
    return [get_onehot_for_index(unique_labels, config.index(x)) for x in labels]

# -*- coding: utf-8 -*-
"""
@created on: 1/23/20,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""


class StringConstants:
    DATA_READ_FAILED = 'Data read failed. Please sure make filepath is correct and file is populated'
    FILEPATH_EMPTY = 'File path is none. Please pass a valid file'
    DATA_SPLIT_FAILED = 'Failed to split data into train and test'
    TRAIN_SUCCESSFUL = 'Successfully trained the model'
    MODEL_SAVE_SUCCESSFUL = 'Successfully saved the trained model under'
    DATA_HAS_NO_CONTENT = 'The data passed is empty. Please provide valid data'
    MODEL_INIT = 'Model initialized successfully'
    PREDICTION_ERROR = 'Encountered an error in prediction'
    PREDICTION_SUCCESSFUL = 'Prediction successful'
    MODEL_NOT_FOUND = 'Trained model not found'


class ModelConstants:
    ONEHOT_CONFIG = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    TRAIN_DATA_PATH = '/data/iris.csv'
    MODEL_SAVEPATH = '/saved_models/model.pk'

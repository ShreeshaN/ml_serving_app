# -*- coding: utf-8 -*-
"""
@created on: 1/23/20,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description: This is the service layer of our app, where all the business logic of training a model is written.
                To be precise: Data loading, data transformations, model training etc

..todo::

"""

from pl_src.classifiers.DecisionTreeClassifier import DecisionTreeClassifier
from pl_src.utils.data_utils import read_csv, data_prep, convert_label_to_onehot
from pl_src.utils.constants import StringConstants, ModelConstants
import os
import logging
from flask import current_app

project_basepath = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def train_decision_tree_service():
    """
    This method calls the DecisionTreeClassifier() on IRIS dataset. Basically trains the classifier to fit to IRIS data
    :return: A string message indicating the result of training.
    """

    current_app.logger.info('Reading Input data')

    # Data path is hardcoded for the sake of simplicity
    data = read_csv(project_basepath + ModelConstants.TRAIN_DATA_PATH)
    if data is None or len(data) == 0:
        raise Exception(StringConstants.DATA_READ_FAILED)
    current_app.logger.info('Data size ' + str(data.shape))

    tr_x, tr_y = data_prep(data, train=True)

    # Data Validation
    if tr_x is None or tr_y is None:
        raise Exception(StringConstants.DATA_SPLIT_FAILED)

    tr_y = convert_label_to_onehot(tr_y)

    current_app.logger.info('Data Transformations complete')

    # Create an instance of the classifier
    model = DecisionTreeClassifier()

    # Train the classifier
    result = model.fit(tr_x, tr_y)
    current_app.logger.info(result)

    # Save the classifier
    model.save(project_basepath + ModelConstants.MODEL_SAVEPATH)
    current_app.logger.info(StringConstants.MODEL_SAVE_SUCCESSFUL + ' ' + ModelConstants.MODEL_SAVEPATH)

    return result

# -*- coding: utf-8 -*-
"""
@created on: 1/23/20,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description: This is the service layer of our app, where all the business logic for serving a model is written.
                To be precise: Model testing, model serving etc

..todo::

"""

from pl_src.utils.constants import StringConstants
from pl_src.classifiers.DecisionTreeClassifier import DecisionTreeClassifier
from pl_src.utils.data_utils import data_prep, convert_onehot_to_label
from pl_src.utils.constants import StringConstants
from flask import current_app
import numpy as np

# Creating this variable during app start-up. A trained model is loaded upon first serve request, then kept in memory
model = None
print('Model placeholder initialised')


def serve_decision_tree_service(test_data):
    """
    This method calls the restores the DecisionTreeClassifier() which is trained on IRIS dataset and runs the
    passed parameter through trained model to get prediction.
    :param test_data: Data for which you need predictions from the decision tree classifier trained on IRIS data
    :return: String encapsulating Prediction for the given data, if data length is 1, else a dictionary of predictions.
            The length of the prediction dict will be equal to input length. Look for results under key: "predictions"
    """
    global model
    if test_data is None:
        raise Exception(StringConstants.DATA_HAS_NO_CONTENT)
    test_data = np.array(test_data)
    current_app.logger.info('Data size ' + str(test_data.shape))

    # Initialise the model for the first request and keep it in memory
    if model is None:
        model = DecisionTreeClassifier(train=False)
        current_app.logger.info(StringConstants.MODEL_INIT)

    test_data = data_prep(test_data, train=False)
    prediction = model.predict(test_data)
    if prediction is None:
        raise Exception(StringConstants.PREDICTION_ERROR)
    current_app.logger.info(StringConstants.PREDICTION_SUCCESSFUL)
    if len(test_data) == 1:
        return 'Prediction for ' + str(test_data) + ' is ' + str(convert_onehot_to_label(prediction))
    else:
        return {"predictions": convert_onehot_to_label(prediction)}

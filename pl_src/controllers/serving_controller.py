# -*- coding: utf-8 -*-
"""
@created on: 1/23/20,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description: This file acts as a Rest end point layer, basically a controller. It will expose all the Rest end points
                required to test an ML classifier or a neural network.
                Suppose we need to test an already trained model, we do it here

..todo::

"""
from flask import Blueprint
from pl_src.service_layer.model_serving_service_layer import serve_decision_tree_service
from pl_src.utils.constants import StringConstants

from flask import request

model_serve_handler = Blueprint('model_serve_handler', __name__)


@model_serve_handler.route('/serve_decision_tree', methods=['GET', 'POST'])
def serve_decision_tree():
    """
    This method acts as a REST end point to serve a trained decision tree classifier. It accepts samples from IRIS dataset for the moment.
    :param test_data: Data for which you need predictions from the decision tree classifier trained on IRIS data
    :return: String encapsulating Prediction for the given data, if data length is 1, else a dictionary of predictions.
            The length of the prediction dict will be equal to input length. Look for results under key: "predictions"
    """
    test_data = request.get_json(force=True)['data']
    if test_data is None:
        raise Exception(StringConstants.DATA_HAS_NO_CONTENT)
    s = serve_decision_tree_service(test_data)
    print('************************')
    print(s)
    print('************************')
    return s

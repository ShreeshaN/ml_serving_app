# -*- coding: utf-8 -*-
"""
@created on: 1/23/20,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description: This file acts as a Rest end point layer, basically a controller. It will expose all the Rest end points
                required to train an ML classifier or a neural network.
                Suppose we need to train and expose a new model, we do it here

..todo::

"""
from flask import Blueprint
from pl_src.service_layer.train_service_layer import train_decision_tree_service

train_handler = Blueprint('train_handler', __name__)


@train_handler.route('/train_decision_tree')
def train_decision_tree():
    """
    This method acts as a REST end point to train a decision tree. At this point, it is hard-coded to use IRIS dataset.
    :return: A string message indicating the result of training.
    """
    return train_decision_tree_service()

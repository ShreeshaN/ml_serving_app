# -*- coding: utf-8 -*-
"""
@created on: 1/23/20,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description: This is the service layer of our name, where all the business logic is written.

..todo::

"""

from pl_src.classifiers.DecisionTreeClassifier import DecisionTreeClassifier


def train_decision_tree_service():
    """
    This method calls the DecisionTreeClassifier() on IRIS dataset. Basically trains the classifier to fit to IRIS data
    :return: A string message indicating the result of training.
    """
    model = DecisionTreeClassifier()
    model.fit()

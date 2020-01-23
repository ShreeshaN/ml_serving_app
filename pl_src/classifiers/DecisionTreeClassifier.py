# -*- coding: utf-8 -*-
"""
@created on: 1/23/20,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description: Decision Tree classifier

..todo:: Implement different cost functions and accuracy metrics suiting business needs.

"""

from pl_src.classifiers.MLModel import MLModel
from sklearn import tree
from pl_src.utils.constants import StringConstants, ModelConstants
from flask import current_app
import os

project_basepath = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


class DecisionTreeClassifier(MLModel):

    def __init__(self, train=True):
        """
        :param train: Bool variable, true if a new instance has to be created, false if a saved instance has to be restored
        Any pre-requisite initialisations required needs to be done here
        """
        super().__init__()
        current_app.logger.info(StringConstants.MODEL_INIT)
        if train:
            self.classifier = tree.DecisionTreeClassifier()
        else:
            self.classifier = self.restore(project_basepath + ModelConstants.MODEL_SAVEPATH)

    def fit(self, x, y):
        """
        Train part of our algorithm
        :param x: Input data, a list of lists
        :param y: Target labels, a list
        :return: A string message indicating the result of training.
        """
        self.classifier.fit(x, y)
        return StringConstants.TRAIN_SUCCESSFUL

    def evaluate(self, x, y):
        """
        This method is to evaluate model on unseen data. To calculate validation accuracy.
        :param x: Input data. The number of features should match the features used in training. A list of lists
        :param y: target labels. Should be a list
        :return:
        """
        pass

    def predict(self, x):
        """
        Used to predict the results using the trained model.
        :param x: Input features. The number of features should match the features used in training
        :return: Predicted label
        """
        return self.classifier.predict(x)

    def cost_fn(self, x, y):
        """
        Optional cost function to evaluate trained model
        :param x: Input data. The number of features should match the features used in training
        :param y: target labels
        :return: Cost value
        """
        pass

    def accuracy(self, x, y):
        """
        Optional metric to evaluate trained model
        :param x: Input data. The number of features should match the features used in training
        :param y: target labels
        :return: Metric value
        """
        pass

    def confusion_matrix(self, x, y):
        """
        Optional metric to evaluate trained model
        :param x: Input data. The number of features should match the features used in training
        :param y: target labels
        :return: Confusion Matrix
        """
        pass

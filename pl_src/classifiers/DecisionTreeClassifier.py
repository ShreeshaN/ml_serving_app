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
from pl_src.utils.string_constants import StringConstants


class DecisionTreeClassifier(MLModel):

    def __init__(self):
        """
        Any pre-requisite initialisations required needs to be done here
        """
        super().__init__()
        print('Initialising Decision Tree')

    def fit(self, x, y):
        """
        Train part of our algorithm
        :param x: Input data
        :param y: Target labels
        :return: A string message indicating the result of training.
        """
        classifier = tree.DecisionTreeClassifier()
        classifier.fit(x, y)
        return StringConstants.TRAIN_SUCCESSFUL

    def evaluate(self, x, y):
        """
        This method is to evaluate model on unseen data. To calculate validation accuracy.
        :param x: Input data. The number of features should match the features used in training
        :param y: target labels
        :return:
        """
        pass

    def predict(self, x):
        """
        Used to predict the results using the trained model.
        :param x: Input features. The number of features should match the features used in training
        :return: Predicted label
        """
        pass

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

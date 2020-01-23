# -*- coding: utf-8 -*-
"""
@created on: 1/23/20,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description: This is an abstract parent class for any ML classifier we write. Any ML classifier should have its own class and
                should inherit properties from this class.

..todo::

"""

from abc import ABC, abstractmethod
import pickle
from pl_src.utils.string_constants import StringConstants


class MLModel(ABC):

    def __init__(self):
        """
        Any pre-requisite initialisations required needs to be done here
        """
        pass

    @abstractmethod
    def fit(self, x, y):
        """
        Train part of our algorithm
        :param x: Input data
        :param y: Target labels
        :return:
        """
        pass

    @abstractmethod
    def evaluate(self, x, y):
        """
        This method is to evaluate model on unseen data. To calculate validation accuracy.
        :param x: Input data. The number of features should match the features used in training
        :param y: target labels
        :return:
        """
        pass

    @abstractmethod
    def predict(self, x):
        """
        Used to predict the results using the trained model.
        :param x: Input features. The number of features should match the features used in training
        :return: Predicted label
        """
        pass

    def save(self, path):
        """
        Save your trained model using this method.
        :param path: Path of the file system where model has to be saved
        :return: None
        """
        if path is None:
            raise Exception(StringConstants.FILEPATH_EMPTY)
        pickle.dump(self, open(path, 'wb'))

    def restore(self, path):
        """
        Restore a already trained model
        :param path: Path of the saved model
        :return: Restored Model
        """
        if path is None:
            raise Exception(StringConstants.FILEPATH_EMPTY)
        return pickle.load(open(path))

    @abstractmethod
    def cost_fn(self, x, y):
        """
        Optional cost function to evaluate trained model
        :param x: Input data. The number of features should match the features used in training
        :param y: target labels
        :return: Cost value
        """
        pass

    @abstractmethod
    def accuracy(self, x, y):
        """
        Optional metric to evaluate trained model
        :param x: Input data. The number of features should match the features used in training
        :param y: target labels
        :return: Metric value
        """
        pass

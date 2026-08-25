"""
This module provides the core class of the wrapshap package.
"""

# Imports 

import numpy as np
import pandas as pd

from shap import summary_plot

from ._internal._wrapshap import shapRFE, shapFFS, shapBFE
from ._internal._wrapshap import get_surrogate_metrics

# ---------------------------------------------------------------------------------------------------------------------
# Public API - These utilities are used by other modules and re-exported.
# ---------------------------------------------------------------------------------------------------------------------


class Wrapshap:
    """
    The Wrapshap base class.

    Attributes:
        X_train (np.array): The original train set as a 2D array.
        X_test (np.array): The original test set as a 2D array.
        y_train (np.array): The original train target as a 1D array.
        y_test (np.array): The original test target as a 1D array.
        shap_train (np.array): The SHAP values for the train set as a 2D array.
        shap_test (np.array): The SHAP values for the test set as a 2D array.
        y_pred_train (np.array): The complex model predictions for the train set as a 1D array.
        y_pred_test (np.array): The complex model predictions for the test set as a 1D array.
        feature_names (list): The feature names.
        task_type (str): The type of task. Accepted values are:
                        - 'regression': for a regression task.
                        - 'classification': for a binary classification task.
    """
    def __init__(
            self,
            X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
            shap_train: np.ndarray, shap_test: np.ndarray, y_pred_train: np.ndarray, y_pred_test: np.ndarray,
            feature_names: list,
            task_type: str
    ):
        
        for array in [X_train, X_test, shap_train, shap_test]:
            if not isinstance(array, np.ndarray) or array.ndim != 2:
                raise ValueError("'X_train, 'X_test', 'shap_train' and 'shap_test' should all be 2D numpy arrays.")
            
        for array in [y_train, y_test, y_pred_train, y_pred_test]:
            if not isinstance(array, np.ndarray) or array.ndim != 1:
                raise ValueError("'y_train', 'y_test', 'y_pred_train' and 'y_pred_test' should all be 1D numpy arrays.")
            
        if X_train.shape != shap_train.shape:
            raise ValueError("'X_train' and 'shap_train' should have the same shape.")
        
        if y_train.shape != y_pred_train.shape:
            raise ValueError("'y_train' and 'y_pred_train' should have the same shape.")
        
        if X_test.shape != shap_test.shape:
            raise ValueError("'X_test' and 'shap_test' should have the same shape.")
        
        if y_test.shape != y_pred_test.shape:
            raise ValueError("'y_test' and 'y_pred_test' should have the same shape.")
        
        if X_train.shape[1] != X_test.shape[1]:
            raise ValueError("'X_train' and 'X_test' should have the same number of columns.")
        
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("'X_train' and 'y_train' should have the same number of instances.")
        
        if isinstance(feature_names, (list, pd.Series)):
            feature_names = [str(f) for f in feature_names]
        else:
            raise TypeError("'feature_names' should be a list of strings.")
        
        if not isinstance(task_type, str) or task_type not in ['regression', 'classification']:
            raise ValueError("'task_type' should be a string with value either 'regression' or 'classification'.")
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.shap_train = shap_train
        self.shap_test = shap_test
        self.y_pred_train = y_pred_train
        self.y_pred_test = y_pred_test
        self.feature_names = feature_names
        self.task_type = task_type


    def shapRFE(self, surrogate_metrics: bool = False, top: int = None):
        """
        Wrapshap implementation of Recursive Feature Elimination (RFE). 

        Parameters:
            surrogate_metrics (bool): Whether the compute and return the surrogate metrics or not.
            top (int): When different than None, instead of performing RFE on all features, perfoms it only on the top 
                        features (the ones with biggest mean absolute SHAP values)

        Returns:
            tuple[list, dict]: A tuple containing the list of ranked features (from most to least important) and a 
                            dictionary with the corresponding performance metrics.
        """

        return shapRFE(
            shap_train=self.shap_train, shap_test=self.shap_test,
            y_train=self.y_train, y_test=self.y_test,
            y_pred_train=self.y_pred_train, y_pred_test=self.y_pred_test,
            feature_names=self.feature_names,
            surrogate_metrics=surrogate_metrics,
            top=top,
            task_type=self.task_type
        )
    

    def shapFFS(self, surrogate_metrics: bool = False, top: int = None, plot:bool = True):
        """
        Wrapshap implementation of Forward Feature Selection (FFS). 

        Parameters:
            surrogate_metrics (bool): Whether the compute and return the surrogate metrics or not.
            top (int): When different than None, instead of performing FFS on all features, perfoms it only on the top 
                        features (the ones with biggest mean absolute SHAP values)

        Returns:
            tuple[list, dict]: A tuple containing the list of ranked features (from most to least important) and a 
                            dictionary with the corresponding performance metrics.
        """

        return shapFFS(
            shap_train=self.shap_train, shap_test=self.shap_test,
            y_train=self.y_train, y_test=self.y_test,
            y_pred_train=self.y_pred_train, y_pred_test=self.y_pred_test,
            feature_names=self.feature_names,
            surrogate_metrics=surrogate_metrics,
            top=top,
            task_type=self.task_type,
            plot=plot
        )
    

    def shapBFE(self, surrogate_metrics: bool = False, top: int = None):
        """
        Wrapshap implementation of Backward Feature Elimination (BFE). 

        Parameters:
            surrogate_metrics (bool): Whether the compute and return the surrogate metrics or not.
            top (int): When different than None, instead of performing BFE on all features, perfoms it only on the top 
                        features (the ones with biggest mean absolute SHAP values)

        Returns:
            tuple[list, dict]: A tuple containing the list of ranked features (from most to least important) and a 
                            dictionary with the corresponding performance metrics.
        """

        return shapBFE(
            shap_train=self.shap_train, shap_test=self.shap_test,
            y_train=self.y_train, y_test=self.y_test,
            y_pred_train=self.y_pred_train, y_pred_test=self.y_pred_test,
            feature_names=self.feature_names,
            surrogate_metrics=surrogate_metrics,
            top=top,
            task_type=self.task_type
        )
    

    def get_surrogate_metrics(self, feature_ranking: list, metrics_dict: dict = {}):
        """
        Iteratively computes the surrogate metrics for subsets from the provided feature ranking.

        Parameters:
            feature_ranking (list): The ranked features.
            metrics_dict (dict): The metrics dictionary to which surrogate metrics will be added.

        Returns:
            dict: A dictionary containing the computed surrogate performance metrics.
        """

        return get_surrogate_metrics(
            shap_train=self.shap_train, shap_test=self.shap_test, y_pred_train=self.y_pred_train,
            y_train=self.y_train, y_test=self.y_test,
            feature_names=self.feature_names,
            feature_ranking=feature_ranking,
            metrics_dict=metrics_dict,
            task_type=self.task_type
        )
    

    def shap_summary_plot(self, return_ranking: bool = False, max_display: int = None):
        """
        Plots a SHAP Summary Plot.

        Parameters:
            return_ranking (bool): Whether or not to return the features ranked by mean absolute SHAP values (ShapAbs).
            max_display (int): How many top features to include in the plot.

        Returns:
            list: If return_ranking is True, the list of ranked features.
        """

        summary_plot(
            shap_values=self.shap_train, 
            features=self.X_train, 
            feature_names=self.feature_names, 
            max_display=max_display
        )

        if return_ranking:
            return [feature for _, feature in sorted(zip(np.sum(np.abs(self.shap_train), axis=0), 
                                                         self.feature_names), reverse=True)]
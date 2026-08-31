"""
This module provides utility functions and convenient classes of LightGBM regressors and classifiers already wrapped
with Wrapshap.
"""

# Imports 

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from functools import wraps
from abc import ABC, abstractmethod
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import r2_score, mean_absolute_error

from ..wrapshap import Wrapshap

# ----------------------------------------------------------------------------------------------------------------------
# Public API - These utilities are used by other modules and re-exported.
# ----------------------------------------------------------------------------------------------------------------------


def wrapshap_required(*attrs):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            for attr in attrs:
                if getattr(self, attr, None) is None:
                    if self.model is None:
                        print('No model found.')
                        print('Fitting model with default parameters..')
                        self.fit()
                        print('Done.')
                    print("Computing SHAP values and predictions..")
                    self.compute_shap_and_predictions()
                    print('Done.')
                    break
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


class WrappedLGB(ABC):
    """
    This base class provides an LightGBM already conveniently wrapped with Wrapshap. This allows you to fit a model and 
    compute its SHAP values in a straigthforward way. If you want to have more control over the model and the SHAP 
    computation, please train your model separately and directly use the Wrapshap class itself.

    Attributes:
        X (np.array): The original data as a 2D array.
        y (np.array): The original target as a 1D array.
        X_train (np.array): The original train set as a 2D array.
        X_test (np.array): The original test set as a 2D array.
        y_train (np.array): The original train target as a 1D array.
        y_test (np.array): The original test target as a 1D array.
        shap_train (np.array): The SHAP values for the train set as a 2D array.
        shap_test (np.array): The SHAP values for the test set as a 2D array.
        y_pred_train (np.array): The complex model predictions for the train set as a 1D array.
        y_pred_test (np.array): The complex model predictions for the test set as a 1D array.
        feature_names (list): The feature names.
        random_state (int): The chosen random state for reproductibility.
        model (lgb.Booster): The LGBM model.
        lgb_params (dict): The LGBM model parameters.
        wrapshap (Wrapshap): The underlying Wrapshap object being called.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, feature_names: list = None, random_state: int = 42, **kwargs):
        """
        Parameters:
            X (np.array): The original data as a 2D array.
            y (np.array): The original target as a 1D array.
            feature_names (list): The feature names.
            random_state (int): The chosen random state for reproductibility.
            **kwargs: Any parameters to add to the LightGBM configuration.
        """

        if type(self) is WrappedLGB:
            raise TypeError(f"{type(self)} is an abstract class and cannot be instanciated directly.")
        
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise ValueError("'X' should be a 2D numpy array.")
        
        if not isinstance(y, np.ndarray) or y.ndim != 1:
            raise ValueError("'y' should be a 1D numpy array.")

        if feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(1, X.shape[1]+1)]
        else:
            if isinstance(feature_names, (list, pd.Series)):
                feature_names = [str(f) for f in feature_names]
            else:
                raise TypeError("'feature_names' should be a list of strings.")
            self.feature_names = feature_names
        
        self.lgb_params = {
                'eta': 0.1,
                'max_depth': 3,
                'seed': random_state
            }
        
        self.lgb_params.update(kwargs)
        
        self.X = X
        self.y = y
        self.random_state = random_state

        self.model = None
        self.shap_train = None
        self.shap_test = None
        self.y_pred_train = None
        self.y_pred_test = None
        self.wrapshap = None


    def compute_shap_and_predictions(self):
        """
        Computes SHAP values and predictions using a LightGBM sklearn model.
        """
    
        if self.model is None:
            raise ValueError("No model was found.")
    
        # --- Predictions ---
        self.y_pred_train = self.model.predict(self.X_train)
        self.y_pred_test = self.model.predict(self.X_test)
    
        # --- SHAP ---
        explainer = shap.TreeExplainer(self.model)
        shap_train = explainer.shap_values(self.X_train)
        shap_test = explainer.shap_values(self.X_test)
    
        # Cas classification binaire (LightGBM retourne une liste)
        if isinstance(shap_train, list):
            shap_train = shap_train[1]
            shap_test = shap_test[1]
    
        self.shap_train = shap_train
        self.shap_test = shap_test
        
    @wrapshap_required('wrapshap')
    def shapRFE(self, surrogate_metrics: bool = False, top: int = None, plot:bool = True):
        """
        Simply calls Wrapshap implementation of Recursive Feature Elimination (RFE). 

        Parameters:
            surrogate_metrics (bool): Whether the compute and return the surrogate metrics or not.
            top (int): When different than None, instead of performing RFE on all features, perfoms it only on the top 
                        features (the ones with biggest mean absolute SHAP values)

        Returns:
            tuple[list, dict]: A tuple containing the list of ranked features (from most to least important) and a 
                            dictionary with the corresponding performance metrics.
        """

        return self.wrapshap.shapRFE(
            surrogate_metrics=surrogate_metrics,
            top=top,
            plot = plot,
        )
    
    
    @wrapshap_required('wrapshap')
    def shapFFS(self, surrogate_metrics: bool = False, top: int = None, plot:bool = True):
        """
        Simply calls Wrapshap implementation of Forward Feature Selection (FFS). 

        Parameters:
            surrogate_metrics (bool): Whether the compute and return the surrogate metrics or not.
            top (int): When different than None, instead of performing FFS on all features, perfoms it only on the top 
                        features (the ones with biggest mean absolute SHAP values)

        Returns:
            tuple[list, dict]: A tuple containing the list of ranked features (from most to least important) and a 
                            dictionary with the corresponding performance metrics.
        """

        return self.wrapshap.shapFFS(
            surrogate_metrics=surrogate_metrics,
            top=top,
            plot = plot,
        )
    

    @wrapshap_required('wrapshap')
    def shapBFE(self, surrogate_metrics: bool = False, top: int = None, plot:bool = True):
        """
        Simply calls Wrapshap implementation of Forward Feature Selection (BFE). 

        Parameters:
            surrogate_metrics (bool): Whether the compute and return the surrogate metrics or not.
            top (int): When different than None, instead of performing BFE on all features, perfoms it only on the top 
                        features (the ones with biggest mean absolute SHAP values)

        Returns:
            tuple[list, dict]: A tuple containing the list of ranked features (from most to least important) and a 
                            dictionary with the corresponding performance metrics.
        """

        return self.wrapshap.shapBFE(
            surrogate_metrics=surrogate_metrics,
            top=top,
            plot=plot,
        )
    

    @wrapshap_required('wrapshap')
    def get_surrogate_metrics(self, feature_ranking: list, metrics_dict: dict = {}):
        """
        Iteratively computes the surrogate metrics for subsets from the provided feature ranking.

        Parameters:
            feature_ranking (list): The ranked features.
            metrics_dict (dict): The metrics dictionary to which surrogate metrics will be added.

        Returns:
            dict: A dictionary containing the computed surrogate performance metrics.
        """

        return self.wrapshap.get_surrogate_metrics(
            feature_ranking=feature_ranking,
            metrics_dict=metrics_dict,
        )
    

    @wrapshap_required('wrapshap')
    def shap_summary_plot(self, return_ranking: bool = False, max_display: int = None):
        """
        Plots a SHAP Summary Plot.

        Parameters:
            return_ranking (bool): Whether or not to return the features ranked by mean absolute SHAP values (ShapAbs).
            max_display (int): How many top features to include in the plot.

        Returns:
            list: If return_ranking is True, the list of ranked features.
        """

        return self.wrapshap.shap_summary_plot(
            return_ranking=return_ranking,
            max_display=max_display
        )

    @abstractmethod
    def plot_performance(self):
        pass


    @abstractmethod
    def fit(self):
        pass


    def _fit(self, eval_size, num_boost_round, early_stopping_rounds, verbose_eval):
        X_train, X_val, y_train, y_val = train_test_split(
            self.X_train,
            self.y_train,
            test_size=eval_size,
            random_state=self.random_state
        )
    
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=self.eval_metric,
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=early_stopping_rounds,
                    verbose=verbose_eval
                )
            ]
        )

class WrappedBinaryClassifierLGB(WrappedLGB):
    """
    This class extends the base WrappedLGB class and provides an LGBoost Binary Classifier already conveniently wrapped 
    with Wrapshap. Objective will be binary logistic. Please see WrappedLGB for more information.
    """

    def __init__(
            self, X: np.ndarray, y: np.ndarray, feature_names: list = None, 
            test_size: float = 0.2, random_state: int = 42, encode_label: bool = True,verbose =-1,
            **kwargs
    ):
        """
        Parameters:
            X (np.array): The original data as a 2D array.
            y (np.array): The original target as a 1D array.
            feature_names (list): The feature names.
            test_size (float): Portion of the data being used as the test set.
            random_state (int): The chosen random state for reproductibility.
            encode_label (bool): Whether or not to encode the labels before fitting.
            **kwargs: Any parameters to add to the LightGBM configuration.
        """

        super().__init__(X, y, feature_names, random_state, **kwargs)

        #self.lgb_params['objective'] = 'binary:logistic'

        if encode_label:
            label_encoder = LabelEncoder()
            self.y = label_encoder.fit_transform(self.y)
        self.lgb_params['verbose']= verbose

        self.model = lgb.LGBMClassifier(
            objective="binary",
            random_state=self.random_state,
            **self.lgb_params        )
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, 
                                                                                test_size=test_size, 
                                                                                random_state=random_state)

    def fit(
            self,metric ='logloss' ,class_imbalanced: bool = False, eval_size: float = 0.2, num_boost_round: int = 10000, 
            early_stopping_rounds: int = 10, verbose_eval: bool = False, threshold_prob: float = 0.5
    ):
        """
        Fits the classifier on training data.

        Parameters:
            scale_pos_weight (bool): Whether or not to apply weights proportional to the class imbalance to mitigate it.
            eval_size (float): Portion of the train set being used as the validation set.
            num_boost_round (int): Maximum number of boosting rounds.
            early_stopping_rounds: (int): Number of rounds with no improvement on validation data before training stops.
            verbose_eval (bool): Whether or not to display validation metrics during training.
            threshold_prob (float): Discriminatory threshold used for classification metrics computation.
        """

        if class_imbalanced:
            self.lgb_params["class_weight"] = "balanced"
        else:
            self.lgb_params.pop("class_weight", None)
            
        self.eval_metric = metric
        super()._fit(eval_size, num_boost_round, early_stopping_rounds, verbose_eval)


        self.plot_performance(threshold_prob=threshold_prob)

    
    def plot_performance(self, threshold_prob: float = 0.5):
        """
        Displays model performance.
    
        Parameters:
            threshold_prob (float): Discriminatory threshold used for classification metrics computation.
        """
    
        if self.model is None:
            raise ValueError("No model was found. Please call .fit()")
    
        # --- Probabilities for positive class ---
        y_train_pred_proba = self.model.predict_proba(self.X_train)[:, 1]
        y_test_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
    
        _plot_classification_metrics(
            self.y_train,
            self.y_test,
            y_train_pred_proba,
            y_test_pred_proba,
            threshold_prob
        )

    
    def compute_shap_and_predictions(self):
        """
        Computes model's SHAP values and predictions and instantiates the underlying Wrapshap object.
        """
        super().compute_shap_and_predictions()
    
        # --- Raw predictions (log-odds), equivalent to output_margin=True ---
        self.y_pred_train = self.model.predict(
            self.X_train, raw_score=True
        ).reshape(-1)
    
        self.y_pred_test = self.model.predict(
            self.X_test, raw_score=True
        ).reshape(-1)
    
        self.wrapshap = Wrapshap(
            X_train=self.X_train,
            X_test=self.X_test,
            y_train=self.y_train,
            y_test=self.y_test,
            shap_train=self.shap_train,
            shap_test=self.shap_test,
            y_pred_train=self.y_pred_train,
            y_pred_test=self.y_pred_test,
            feature_names=self.feature_names,
            task_type="classification"
        )

class WrappedRegressorLGB(WrappedLGB):
    """
    This class extends the base WrappedLGB class and provides an LightGBM Regressor already conveniently wrapped 
    with Wrapshap. Objective will be squared error. Please see WrappedLGB for more information.
    """

    def __init__(
            self, X: np.ndarray, y: np.ndarray, feature_names: list = None, 
            test_size: float = 0.2, random_state: int = 42, verbose = -1,
            **kwargs
    ):
        """
        Parameters:
            X (np.array): The original data as a 2D array.
            y (np.array): The original target as a 1D array.
            feature_names (list): The feature names.
            test_size (float): Portion of the data being used as the test set.
            random_state (int): The chosen random state for reproductibility.
            **kwargs: Any parameters to add to the LightGBM configuration.
        """

        super().__init__(X, y, feature_names, random_state, **kwargs)
        self.lgb_params['verbose']= verbose
        self.model = lgb.LGBMRegressor(
            random_state=self.random_state,
            **self.lgb_params        )
            
        #self.lgb_params['objective'] = 'reg:squarederror'

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, 
                                                                                test_size=test_size, 
                                                                                random_state=random_state)

    def fit(
            self, metric = 'mae',eval_size: float = 0.2, num_boost_round: int = 10000, 
            early_stopping_rounds: int = 10, verbose_eval: bool = False
    ):
        """
        Fits the classifier on training data.

        Parameters:
            eval_size (float): Portion of the train set being used as the validation set.
            num_boost_round (int): Maximum number of boosting rounds.
            early_stopping_rounds: (int): Number of rounds with no improvement on validation data before training stops.
            verbose_eval (bool): Whether or not to display validation metrics during training.
            threshold_prob (float): Discriminatory threshold used for classification metrics computation.
        """
        self.eval_metric = metric        
        super()._fit(eval_size, num_boost_round, early_stopping_rounds, verbose_eval)
        self.plot_performance()

    
    def plot_performance(self):
        """
        Displays model performance.
        """
    
        if self.model is None:
            raise ValueError("No model was found. Please call .fit()")
    
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
    
        _plot_regression_metrics(
            self.y_train,
            self.y_test,
            y_train_pred,
            y_test_pred
        )
    
    def compute_shap_and_predictions(self):
        """
        Computes model's SHAP values and predictions and instantiates the underlying Wrapshap object.
        """
    
        super().compute_shap_and_predictions()
    
        self.y_pred_train = self.model.predict(self.X_train).reshape(-1)
        self.y_pred_test = self.model.predict(self.X_test).reshape(-1)
    
        self.wrapshap = Wrapshap(
            X_train=self.X_train,
            X_test=self.X_test,
            y_train=self.y_train,
            y_test=self.y_test,
            shap_train=self.shap_train,
            shap_test=self.shap_test,
            y_pred_train=self.y_pred_train,
            y_pred_test=self.y_pred_test,
            feature_names=self.feature_names,
            task_type="regression"
        )

# ---------------------------------------------------------------------------------------------------------------------
# Internal Functions - Below are utilities intended only for internal use.
# ---------------------------------------------------------------------------------------------------------------------


def _plot_classification_metrics(y_train, y_test, y_train_pred_proba, y_test_pred_proba, threshold_prob=0.5):
    fig = plt.figure(figsize=(15, 15))

    grid = plt.GridSpec(4, 2, wspace=0.4, hspace=0.3)

    y_train_pred = (y_train_pred_proba >= threshold_prob).astype(int)
    y_test_pred = (y_test_pred_proba >= threshold_prob).astype(int)

    scores = {
        'Accuracy': (accuracy_score(y_train, y_train_pred), accuracy_score(y_test, y_test_pred)),
        'Precision': (precision_score(y_train, y_train_pred), precision_score(y_test, y_test_pred)),
        'Recall': (recall_score(y_train, y_train_pred), recall_score(y_test, y_test_pred)),
        'F1': (f1_score(y_train, y_train_pred), f1_score(y_test, y_test_pred))
    }
    
    ax1 = fig.add_subplot(grid[0:2, :])
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred_proba)
    roc_auc_train = auc(fpr_train, tpr_train)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred_proba)
    roc_auc_test = auc(fpr_test, tpr_test)
    ax1.plot(fpr_train, tpr_train, label=f'Train ROC (AUC = {roc_auc_train:.2f})')
    ax1.plot(fpr_test, tpr_test, label=f'Test ROC (AUC = {roc_auc_test:.2f})')
    ax1.plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title(f'ROC curve (LightGBM)')

    legend_handles = [mlines.Line2D([], [], color='blue', marker='_', linestyle='-', markersize=10, label=f'Train ROC (AUC = {roc_auc_train:.2f})'),
                    mlines.Line2D([], [], color='orange', marker='_', linestyle='-', markersize=10, label=f'Test ROC (AUC = {roc_auc_test:.2f})')]
    for score_name, (train_score, test_score) in scores.items():
        dummy_line = mlines.Line2D([], [], color='none', marker='_', linestyle='-', markersize=0,
                                label=f'{score_name}: train {train_score:.2f} / test {test_score:.2f}')
        legend_handles.append(dummy_line)
    ax1.legend(handles=legend_handles, loc="lower right")

    ax2 = fig.add_subplot(grid[2, 0])
    cf_matrix_train = confusion_matrix(y_train, y_train_pred)
    sns.heatmap(cf_matrix_train, annot=True, cmap='Blues', fmt='g', ax=ax2)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    ax2.set_title('Confusion Matrix (Train)')

    ax3 = fig.add_subplot(grid[2, 1])
    cf_matrix_test = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cf_matrix_test, annot=True, cmap='Blues', fmt='g', ax=ax3)
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('True')
    ax3.set_title('Confusion Matrix (Test)')
    plt.show()


def _plot_regression_metrics(y_train, y_test, y_train_pred, y_test_pred):
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    _, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].scatter(y_train, y_train_pred, alpha=0.3)
    ax[0].set_xlabel('True Values')
    ax[0].set_ylabel('Predictions')
    ax[0].set_title(f'Training Set: R2 = {train_r2:.2f}, MAE = {train_mae:.2f}')

    ax[1].scatter(y_test, y_test_pred, alpha=0.3)
    ax[1].set_xlabel('True Values')
    ax[1].set_ylabel('Predictions')
    ax[1].set_title(f'Test Set: R2 = {test_r2:.2f}, MAE = {test_mae:.2f}')

    plt.show()
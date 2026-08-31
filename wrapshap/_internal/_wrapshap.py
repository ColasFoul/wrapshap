"""
This module provides utility functions for the wrapshap package.
It includes both functions intended for internal use within the package
and functions that are exported for use in public modules.
"""

# Imports

import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.special import expit
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------
# Public API - These utilities are used by other modules and re-exported.
# ----------------------------------------------------------------------------------------------------------------------


def shapRFE(
        shap_train: np.ndarray, shap_test: np.ndarray, y_pred_train: np.ndarray, y_pred_test: np.ndarray,
        y_train: np.ndarray, y_test: np.ndarray,
        feature_names: list,
        surrogate_metrics: bool = False,
        top: int = None,
        task_type: str = 'classification',
        plot : bool = True,
) -> tuple:
    """
    Wrapshap implementation of Recursive Feature Elimination (RFE). 

    Parameters:
        shap_train (np.array): The SHAP values for the train set as a 2D array.
        shap_test (np.array): The SHAP values for the test set as a 2D array.
        y_pred_train (np.array): The complex model predictions for the train set as a 1D array.
        y_pred_test (np.array): The complex model predictions for the test set as a 1D array.
        feature_names (list): The feature names.
        surrogate_metrics (bool): Whether the compute and return the surrogate metrics or not.
        top (int): When different than None, instead of performing RFE on all features, perfoms it only on the top 
                    features (the ones with biggest mean absolute SHAP values)
        task_type (str): The type of task. Accepted values are:
                        - 'regression': for a regression task.
                        - 'classification': for a binary classification task.

    Returns:
        tuple[list, dict]: A tuple containing the list of ranked features (from most to least important) and a 
                        dictionary with the corresponding performance metrics.
    """

    if top is not None:
        shap_train, shap_test, feature_names = _select_top(top, shap_train, shap_test, feature_names)

    y_pred_train = y_pred_train.squeeze()
    remaining_features = feature_names.copy()
    removed_features = []
    results = {}

    scaler = StandardScaler()
    shap_train_normalized = scaler.fit_transform(shap_train)
    shap_test_normalized = scaler.transform(shap_test)

    _progress_bar(0, len(feature_names), mode='ascending')

    while remaining_features:

        model = LinearRegression()
        feature_indices = sorted([feature_names.index(f) for f in remaining_features])
        shap_train_tmp = shap_train_normalized[:, feature_indices].copy()
        shap_test_tmp = shap_test_normalized[:, feature_indices].copy()

        model.fit(shap_train_tmp, y_pred_train)

        lnr_predictions_train = model.predict(shap_train_tmp).squeeze()
        lnr_predictions_test = model.predict(shap_test_tmp).squeeze()

        metrics_train = _calculate_metrics(y_train, lnr_predictions_train, y_pred_train, 
                                           task_type, surrogate=surrogate_metrics)
        metrics_test = _calculate_metrics(y_test, lnr_predictions_test, y_pred_test, 
                                          task_type, surrogate=surrogate_metrics)

        results[len(remaining_features)] = {}
        for key in metrics_test:
            results[len(remaining_features)][f'test_{key}'] = metrics_test[key]
        for key in metrics_train:
            results[len(remaining_features)][f'train_{key}'] = metrics_train[key]

        min_importance_index = np.argmin(np.abs(model.coef_))
        removed_feature = remaining_features.pop(min_importance_index)
        removed_features.append(removed_feature)
        
        _progress_bar(len(removed_features), len(feature_names), mode='ascending')

    removed_features.reverse()
    
    if plot :
        _plot_performance(results)
    return removed_features, results


def shapFFS(
        shap_train: np.ndarray, shap_test: np.ndarray, y_pred_train: np.ndarray, y_pred_test: np.ndarray,
        y_train: np.ndarray, y_test: np.ndarray,
        feature_names: list,
        surrogate_metrics: bool = False,
        top: int = None,
        task_type: str = 'classification',
        plot: bool = True
) -> tuple:
    """
    Wrapshap implementation of Forward Feature Selection (FFS). 

    Parameters:
        shap_train (np.array): The SHAP values for the train set as a 2D array.
        shap_test (np.array): The SHAP values for the test set as a 2D array.
        y_pred_train (np.array): The complex model predictions for the train set as a 1D array.
        y_pred_test (np.array): The complex model predictions for the test set as a 1D array.
        feature_names (list): The feature names.
        surrogate_metrics (bool): Whether the compute and return the surrogate metrics or not.
        top (int): When different than None, instead of performing FFS on all features, perfoms it only on the top 
                    features (the ones with biggest mean absolute SHAP values)
        task_type (str): The type of task. Accepted values are:
                        - 'regression': for a regression task.
                        - 'classification': for a binary classification task.

    Returns:
        tuple[list, dict]: A tuple containing the list of ranked features (from most to least important) and a 
                        dictionary with the corresponding performance metrics.
    """

    if top is not None:
        shap_train, shap_test, feature_names = _select_top(top, shap_train, shap_test, feature_names)

    selected_features = []
    n_features = len(feature_names)
    results = {}

    _progress_bar(0, len(feature_names), mode='ascending')

    while len(selected_features) < n_features:
        best_score = float('-inf')
        best_feature = None

        for feature in feature_names:
            if feature not in selected_features:
                current_features_indices = sorted([feature_names.index(f) for f in selected_features + [feature]])

                model = LinearRegression()
                shap_train_tmp = shap_train[:, current_features_indices].copy()
                shap_test_tmp = shap_test[:, current_features_indices].copy()

                model.fit(shap_train_tmp, y_pred_train)

                lnr_predictions_train = model.predict(shap_train_tmp)
                lnr_predictions_test = model.predict(shap_test_tmp)

                metrics_train = _calculate_metrics(y_train, lnr_predictions_train, y_pred_train, 
                                                   task_type, surrogate=False)
                metrics_test = _calculate_metrics(y_test, lnr_predictions_test, y_pred_test, 
                                                  task_type, surrogate=False)

                if metrics_train['pred_r2'] > best_score:
                    best_score = metrics_train['pred_r2']
                    best_feature = feature
                    best_metrics = {}
                    for key in metrics_test:
                        best_metrics[f'test_{key}'] = metrics_test[key]
                    for key in metrics_train:
                        best_metrics[f'train_{key}'] = metrics_train[key]


        selected_features.append(best_feature)
        results[len(selected_features)] = best_metrics

        _progress_bar(len(selected_features), len(feature_names), mode='ascending')

    if surrogate_metrics:
        results = get_surrogate_metrics(
            shap_train=shap_train, shap_test=shap_test, y_pred_train=y_pred_train,
            y_train=y_train, y_test=y_test,
            feature_names=feature_names,
            feature_ranking=selected_features,
            metrics_dict=results,
            top=top,
            task_type=task_type
        )
    if plot :
        _plot_performance(results)

    return selected_features, results


def shapBFE(
        shap_train: np.ndarray, shap_test: np.ndarray, y_pred_train: np.ndarray, y_pred_test: np.ndarray,
        y_train: np.ndarray, y_test: np.ndarray,
        feature_names: list,
        surrogate_metrics: bool = False,
        top: int = None,
        task_type: str = 'classification',
        plot: bool = True
) -> tuple:
    """
    Wrapshap implementation of Backward Feature Elimination (BFE). 

    Parameters:
        shap_train (np.array): The SHAP values for the train set as a 2D array.
        shap_test (np.array): The SHAP values for the test set as a 2D array.
        y_pred_train (np.array): The complex model predictions for the train set as a 1D array.
        y_pred_test (np.array): The complex model predictions for the test set as a 1D array.
        feature_names (list): The feature names.
        surrogate_metrics (bool): Whether the compute and return the surrogate metrics or not.
        top (int): When different than None, instead of performing BFE on all features, perfoms it only on the top 
                    features (the ones with biggest mean absolute SHAP values)
        task_type (str): The type of task. Accepted values are:
                        - 'regression': for a regression task.
                        - 'classification': for a binary classification task.

    Returns:
        tuple[list, dict]: A tuple containing the list of ranked features (from most to least important) and a 
                        dictionary with the corresponding performance metrics.
    """

    if top is not None:
        shap_train, shap_test, feature_names = _select_top(top, shap_train, shap_test, feature_names)

    selected_features = feature_names.copy()
    removed_features = []
    results = {}

    _progress_bar(0, len(feature_names), mode='ascending')

    model = LinearRegression()
    model.fit(shap_train, y_pred_train)

    lnr_predictions_train = model.predict(shap_train)
    lnr_predictions_test = model.predict(shap_test)

    metrics_train = _calculate_metrics(y_train, lnr_predictions_train, y_pred_train, task_type)
    metrics_test = _calculate_metrics(y_test, lnr_predictions_test, y_pred_test, task_type)

    results[len(selected_features)] = {}
    for key in metrics_test:
        results[len(selected_features)][f'test_{key}'] = metrics_test[key]
    for key in metrics_train:
        results[len(selected_features)][f'train_{key}'] = metrics_train[key]
    
    _progress_bar(1, len(feature_names), mode='ascending')

    while len(selected_features) > 1:
        worst_score = float('-inf')
        worst_feature = None

        for feature in selected_features:
            current_features_indices = sorted([feature_names.index(f) for f in selected_features if f != feature])

            model = LinearRegression()
            shap_train_tmp = shap_train[:, current_features_indices].copy()
            shap_test_tmp = shap_test[:, current_features_indices].copy()

            model.fit(shap_train_tmp, y_pred_train)

            lnr_predictions_train = model.predict(shap_train_tmp)
            lnr_predictions_test = model.predict(shap_test_tmp)

            metrics_train = _calculate_metrics(y_train, lnr_predictions_train, y_pred_train, task_type, surrogate=False)
            metrics_test = _calculate_metrics(y_test, lnr_predictions_test, y_pred_test, task_type, surrogate=False)

            if metrics_train['pred_r2'] > worst_score:
                worst_score = metrics_train['pred_r2']
                worst_feature = feature
                worst_metrics = {}
                for key in metrics_test:
                    worst_metrics[f'test_{key}'] = metrics_test[key]
                for key in metrics_train:
                    worst_metrics[f'train_{key}'] = metrics_train[key]

        selected_features.remove(worst_feature)
        removed_features.append(worst_feature)
        results[len(selected_features)] = worst_metrics

        _progress_bar(len(removed_features)+1, len(feature_names), mode='ascending')

    removed_features.append(selected_features[0])
    removed_features.reverse()

    if surrogate_metrics:
        results = get_surrogate_metrics(
            shap_train=shap_train, shap_test=shap_test, y_pred_train=y_pred_train,
            y_train=y_train, y_test=y_test,
            feature_names=feature_names,
            feature_ranking=selected_features,
            metrics_dict=results,
            top=top,
            task_type=task_type
        )
    if plot :
        _plot_performance(results)

    return removed_features, results


def get_surrogate_metrics(
        shap_train: np.ndarray, shap_test: np.ndarray, y_pred_train: np.ndarray,
        y_train: np.ndarray, y_test: np.ndarray,
        feature_names: list,
        feature_ranking: list,
        metrics_dict: dict = {},
        task_type: str = 'classification'
) -> dict:
    """
    Iteratively computes the surrogate metrics for subsets from the provided feature ranking.

    Parameters:
        shap_train (np.array): The SHAP values for the train set as a 2D array.
        shap_test (np.array): The SHAP values for the test set as a 2D array.
        y_pred_train (np.array): The complex model predictions for the train set as a 1D array.
        feature_names (list): The feature names.
        feature_ranking (list): The ranked features.
        metrics_dict (dict): The metrics dictionary to which surrogate metrics will be added.
        task_type (str): The type of task. Accepted values are:
                        - 'regression': for a regression task.
                        - 'classification': for a binary classification task.

    Returns:
        dict: A dictionary containing the computed surrogate performance metrics.
    """
    
    _progress_bar(0, len(feature_ranking), mode='ascending', suffix='Complete (surrogate metrics computation)')
    for i in range(1, len(feature_ranking)+1):

        model = LinearRegression()
        feature_indices = sorted([feature_names.index(f) for f in feature_ranking[:i]])
        shap_train_tmp = shap_train[:, feature_indices].copy()
        shap_test_tmp = shap_test[:, feature_indices].copy()

        model.fit(shap_train_tmp, y_pred_train)

        lnr_predictions_train = model.predict(shap_train_tmp).squeeze()
        lnr_predictions_test = model.predict(shap_test_tmp).squeeze()

        metrics_train = _surrogate_metrics(y_train, lnr_predictions_train, task_type).copy()
        metrics_test = _surrogate_metrics(y_test, lnr_predictions_test, task_type).copy()

        if i not in metrics_dict:
            metrics_dict[i] = {}
        for key in metrics_test:
            metrics_dict[i][f'test_{key}'] = metrics_test[key]
        for key in metrics_train:
            metrics_dict[i][f'train_{key}'] = metrics_train[key]
        _progress_bar(i, len(feature_ranking), mode='ascending', suffix='Complete (surrogate metrics computation)')
    
    return metrics_dict


# ---------------------------------------------------------------------------------------------------------------------
# Internal Functions - Below are utilities intended only for internal use.
# ---------------------------------------------------------------------------------------------------------------------


def _select_top(top, shap_train, shap_test, features):
    feature_importances = np.sum(np.abs(shap_train), axis=0)
    features_sorted_by_importance = sorted(zip(feature_importances, features), reverse=True)
    top_n_features = [feature for _, feature in features_sorted_by_importance[:top]]
    features_filtered = [fn for fn in features if fn in top_n_features]

    columns_to_keep = [features.index(fn) for fn in features_filtered]
    shap_train_filtered = shap_train[:, columns_to_keep]
    shaps_test_filtered = shap_test[:, columns_to_keep]

    return shap_train_filtered, shaps_test_filtered, features_filtered


def _progress_bar(iteration, total, mode='ascending', prefix='Progress:', suffix='Complete', 
                  length=50, fill='█', print_end="\r"):

    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    if mode == 'ascending':
        iteration_display = f'{iteration}/{total}'
    elif mode == 'descending':
        iteration_display = f'{total - iteration}/{total}'
    print(f'\r{prefix} |{bar}| {iteration_display} {suffix}', end=print_end)
    if iteration == total: 
        print()


def _surrogate_metrics(y_true, lnr_predictions, task_type, metrics={}):

    if task_type=='classification':
        binary_predictions_reconstructed = expit(lnr_predictions)
        metrics['accuracy'] = accuracy_score(y_true, binary_predictions_reconstructed.round())
        metrics['precision'] = precision_score(y_true, binary_predictions_reconstructed.round())
        metrics['recall'] = recall_score(y_true, binary_predictions_reconstructed.round())
        metrics['f1'] = f1_score(y_true, binary_predictions_reconstructed.round())
        metrics['roc_auc'] = roc_auc_score(y_true, binary_predictions_reconstructed)
    
    elif task_type=='regression':
        metrics['r2'] = r2_score(y_true, lnr_predictions)
        metrics['mae'] = mean_absolute_error(y_true, lnr_predictions)


    return metrics


def _calculate_metrics(y_true, lnr_predictions, model_predictions, task_type, surrogate=False):

    metrics = {
        'pred_r2': r2_score(model_predictions, lnr_predictions),
        'pred_mae': mean_absolute_error(model_predictions, lnr_predictions)
    }

    if surrogate:
        metrics = _surrogate_metrics(y_true, lnr_predictions, task_type, metrics)
    
    return metrics

def _plot_performance(perfs):
    n_features_list = []
    r2_list         = []
    
    for n_feat, values in perfs.items():
        n_features_list.append(int(n_feat))
        for metric, value in values.items():
            if metric == 'test_pred_r2':
                r2_list.append(value)
    
    # Trier par nombre de features
    order        = np.argsort(n_features_list)
    n_features   = np.array(n_features_list)[order]
    r2_scores    = np.array(r2_list)[order]
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(n_features, r2_scores,
            color='steelblue', linewidth=2,
            marker='o', markersize=5, zorder=3)
    ax.fill_between(n_features, r2_scores, alpha=0.08, color='steelblue')
    
    
    # Point max (pour référence)
    ax.scatter(n_features[np.argmax(r2_scores)], r2_scores.max(),
               color='gray', zorder=4, s=60, marker='D',
               label=f'Max R²={r2_scores.max():.3f} ({n_features[np.argmax(r2_scores)]} features)')
    
    ax.set_xlabel('Number of selected features', fontsize=12)
    ax.set_ylabel('Test R²', fontsize=12)
    ax.set_title('Feature selection — performance vs. parsimony trade-off',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(n_features)
    ax.set_xticklabels(n_features, rotation=45 if len(n_features) > 10 else 0)
    ax.tick_params(labelsize=9)
    
    plt.tight_layout()
    plt.show()
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
from time import time

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
        time_info : bool = False
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
    start_time = time()
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
            
        results[len(remaining_features)]['time']=time() - start_time

        min_importance_index = np.argmin(np.abs(model.coef_))
        removed_feature = remaining_features.pop(min_importance_index)
        removed_features.append(removed_feature)
        
        _progress_bar(len(removed_features), len(feature_names), mode='ascending')

    removed_features.reverse()
    
    if plot :
        _plot_performance(results, time_info)
    return removed_features, results


def shapFFS(
        shap_train: np.ndarray, shap_test: np.ndarray, y_pred_train: np.ndarray, y_pred_test: np.ndarray,
        y_train: np.ndarray, y_test: np.ndarray,
        feature_names: list,
        surrogate_metrics: bool = False,
        top: int = None,
        task_type: str = 'classification',
        plot: bool = True,
        time_info: bool = True
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
    start_time = time()

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
                    best_metrics['time']=time() - start_time


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
        _plot_performance(results, time_info)

    return selected_features, results


def shapBFE(
        shap_train: np.ndarray, shap_test: np.ndarray, y_pred_train: np.ndarray, y_pred_test: np.ndarray,
        y_train: np.ndarray, y_test: np.ndarray,
        feature_names: list,
        surrogate_metrics: bool = False,
        top: int = None,
        task_type: str = 'classification',
        plot: bool = True,
        time_info: bool = False
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
    start_time = time()

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
    results[len(selected_features)]['time'] = time() - start_time

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
                worst_metrics['time'] = time() - start_time


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
        _plot_performance(results, time_info)

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
    start_time = time()
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
            metrics_dict[i]['time'] = time() - start_time

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

def _plot_performance(perfs, time_info = True):
    n_features_list = []
    r2_list         = []
    time_list       = []
    
    alpha = 1
    beta= 0.2
    
    
    for n_feat, values in perfs.items():
        n_features_list.append(int(n_feat))
        for metric, value in values.items():
            if metric == 'test_pred_r2':
                r2_list.append(value)
            if metric == 'time':
                time_list.append(value)
    
    # Trier par nombre de features
    order        = np.argsort(n_features_list)
    n_features   = np.array(n_features_list)[order]
    r2_scores    = np.array(r2_list)[order]
    time_scores  = np.array(time_list)[order]
        
    def normalize(x):
        x = np.array(x, dtype=float)
        return (x - x.min()) / (x.max() - x.min() + 1e-9)
    
    #r2_norm = normalize(r2_scores)          # on veut maximiser -> reste tel quel
    nf_norm = normalize(n_features)         # on veut minimiser
    time_norm = normalize(time_scores)      # on veut minimiser
    
    fig, ax = plt.subplots(figsize=(8, 4))
    fs_scores = []
    for nf, r2, t in zip(nf_norm, r2_scores, time_norm):
        fs_scores.append((r2*alpha - beta*nf))
    
    fs_scores = np.array(fs_scores)
    # -------------------------------------------------------
    # AXE 1 — R²
    # -------------------------------------------------------
    
    ax.plot(n_features, r2_scores,
            color='steelblue', linewidth=2,
            marker='o', markersize=5, zorder=3, label='Test R²', )
    ax.fill_between(n_features, r2_scores, alpha=0.08, color='steelblue')
    
    best_idx = np.argmax(fs_scores)
    ax.scatter(n_features[best_idx], r2_scores[best_idx],
               color='steelblue', zorder=4, s=60, marker='D',
               label=f'FS_score={fs_scores[best_idx]:.3f} ({n_features[best_idx]} features)')
    
    ax.set_xlabel('Number of selected features', fontsize=12)
    ax.set_ylabel('Test R²', fontsize=12, color='black')
    ax.tick_params(axis='y', labelcolor='black', labelsize = 8)
    ax.spines['top'].set_visible(False)
    
    # -------------------------------------------------------
    # AXE 2 — Computation time
    # -------------------------------------------------------
    if time_info :
        ax2 = ax.twinx()  # partage le même axe x
        ax2.plot(n_features, time_scores,
                 color='coral', linewidth=2, linestyle='--',
                 marker='s', markersize=4, zorder=2, label='Cumulative computation time')
        
        ax2.set_ylabel('Cumulative computation time (s)', fontsize=12, color='black')
        ax2.tick_params(axis='y', labelcolor='black', labelsize = 8)
        ax2.spines['top'].set_visible(False)
    
    # -------------------------------------------------------
    # LÉGENDE COMBINÉE
    # -------------------------------------------------------
    
    lines1, labels1 = ax.get_legend_handles_labels()
    if time_info == True :
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize = 10, framealpha=0.5, loc ='lower right' )
    else :
        ax.legend(lines1, labels1, fontsize = 10, framealpha=0.5, loc ='lower right' )
    
    ax.set_title('Feature selection — performance vs. parsimony trade-off',
                 fontsize=12)
    if len(n_features) > 25:
        ax.set_xticks(n_features[::2])
        ax.set_xticklabels(n_features[::2], rotation=45, fontsize=10)
    if len(n_features) <= 25:
        ax.set_xticks(n_features)
        ax.set_xticklabels(n_features, rotation=45, fontsize=10)
    ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.show()
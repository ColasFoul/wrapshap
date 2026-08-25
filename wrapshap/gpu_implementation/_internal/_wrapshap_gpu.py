"""
This module provides utility functions for GPU implementation of the wrapshap package.
It includes both functions intended for internal use within the package
and functions that are exported for use in public modules.
"""

# Imports

import itertools
import numpy as np
import tensorflow as tf
from graphviz import Digraph
import uuid
from IPython.display import display, Image
from PIL import Image as PILImage
import io
import matplotlib.cm as cm


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from ..._internal._wrapshap import _progress_bar

# ----------------------------------------------------------------------------------------------------------------------
# Public API - These utilities are used by other modules and re-exported.
# ----------------------------------------------------------------------------------------------------------------------

def shapFFS(shap_train: np.ndarray, shap_test: np.ndarray, y_pred_train: np.ndarray, y_pred_test: np.ndarray, feature_names: list, top: int = None):
    """
    Wrapshap GPU implementation of Forward Feature Selection (FFS). 

    Parameters:
        shap_train (np.array): The SHAP values for the train set as a 2D array.
        y_pred_train (np.array): The complex model predictions for the train set as a 1D array.
        feature_names (list): The feature names.
        top (int): When different than None, instead of performing FFS on all features, perfoms it only on the top 
                    features (the ones with biggest mean absolute SHAP values)

    Returns:
        tuple[list, dict]: A tuple containing the list of ranked features (from most to least important) and a 
                        dictionary with the corresponding performance metrics.
    """

    if top is not None:
        shap_train, shap_test, feature_names = _select_top(top, shap_train, shap_test, feature_names)

    y_pred_train = y_pred_train.reshape(-1,1)
    if y_pred_test is not None:
        y_pred_test = y_pred_test.reshape(-1, 1)
    selected_features = []
    n_samples = shap_train.shape[0]
    n_features = len(feature_names)
    results = {}

    _progress_bar(0, len(feature_names))

    while len(selected_features) < n_features:
        best_score = float('-inf')
        best_feature = None
        best_metrics = {}

        batch_combinations = [selected_features + [feature] for feature in feature_names 
                              if feature not in selected_features]

        X_batches = [shap_train[:, [feature_names.index(f) for f in comb]] for comb in batch_combinations]
        y_batches = [y_pred_train for _ in batch_combinations]

        X_batches_tensor = tf.constant(np.stack(X_batches, axis=0), dtype=tf.float32)
        y_batches_tensor = tf.constant(np.stack(y_batches, axis=0), dtype=tf.float32)
        predictions_batches, weights_batches = _process_batch(X_batches_tensor, y_batches_tensor, n_samples)
        batch_r2_scores = _calculate_r2(y_batches_tensor, predictions_batches)
        
        candidate_features = [f for f in feature_names if f not in selected_features]
        for idx, feature in enumerate(candidate_features):
            if batch_r2_scores[idx].numpy() > best_score:
                best_score = batch_r2_scores[idx].numpy()
                best_feature = feature
                best_idx = idx
                best_metrics = {'train_pred_r2': best_score[0]}
        
        # --- évaluation test : on réutilise les poids déjà fit sur train pour la combinaison gagnante ---
        if shap_test is not None and y_pred_test is not None:
            best_comb = selected_features + [best_feature]
            best_weights = weights_batches[best_idx:best_idx+1]  # poids de la combinaison gagnante, shape (1, k+1, 1)
            X_test_best = shap_test[:, [feature_names.index(f) for f in best_comb]]
            X_test_tensor = tf.constant(X_test_best[np.newaxis, ...], dtype=tf.float32)
            y_test_tensor = tf.constant(y_pred_test[np.newaxis, ...], dtype=tf.float32)
            test_predictions = _predict_batch(X_test_tensor, best_weights, shap_test.shape[0])
            test_r2_score = _calculate_r2(y_test_tensor, test_predictions)
            best_metrics['test_pred_r2'] = test_r2_score[0].numpy()[0]

        selected_features.append(best_feature)
        results[len(selected_features)] = best_metrics

        _progress_bar(len(selected_features), len(feature_names))

    return selected_features, results


def shapBFE(shap_train: np.ndarray, shap_test: np.ndarray, y_pred_train: np.ndarray, y_pred_test: np.ndarray, 
            feature_names: list, top: int = None):
    """
    Wrapshap GPU implementation of Backward Feature Elimination (BFE). 
    Parameters:
        shap_train (np.array): The SHAP values for the train set as a 2D array.
        shap_test (np.array): The SHAP values for the test set as a 2D array.
        y_pred_train (np.array): The complex model predictions for the train set as a 1D array.
        y_pred_test (np.array): The complex model predictions for the test set as a 1D array.
        feature_names (list): The feature names.
        top (int): When different than None, instead of performing BFE on all features, performs it only on the top 
                    features (the ones with biggest mean absolute SHAP values)
    Returns:
        tuple[list, dict]: A tuple containing the list of ranked features (from most to least important) and a 
                        dictionary with the corresponding performance metrics.
    """
    if top is not None:
        shap_train, shap_test, feature_names = _select_top(top, shap_train, shap_test, feature_names)
    _progress_bar(0, len(feature_names))
    selected_features = feature_names.copy()
    removed_features = []
    results = {}
    n_samples = shap_train.shape[0]
    y_pred_train = y_pred_train.reshape(-1, 1)
    if y_pred_test is not None:
        y_pred_test = y_pred_test.reshape(-1, 1)

    # --- étape initiale : toutes les variables ---
    X_full = shap_train[:, [feature_names.index(f) for f in selected_features]]
    y_full = y_pred_train

    X_full_tensor = tf.constant(X_full.reshape(1, n_samples, -1), dtype=tf.float32)
    y_full_tensor = tf.constant(y_full.reshape(1, n_samples, -1), dtype=tf.float32)
    predictions_full, weights_full = _process_batch(X_full_tensor, y_full_tensor, n_samples)

    full_r2_score = _calculate_r2(y_full_tensor, predictions_full)
    _progress_bar(1, len(feature_names))

    full_metrics = {'train_pred_r2': full_r2_score.numpy()[0][0]}
    if shap_test is not None and y_pred_test is not None:
        X_test_full = shap_test[:, [feature_names.index(f) for f in selected_features]]
        X_test_full_tensor = tf.constant(X_test_full.reshape(1, shap_test.shape[0], -1), dtype=tf.float32)
        y_test_full_tensor = tf.constant(y_pred_test.reshape(1, shap_test.shape[0], -1), dtype=tf.float32)
        test_predictions_full = _predict_batch(X_test_full_tensor, weights_full, shap_test.shape[0])
        full_metrics['test_pred_r2'] = _calculate_r2(y_test_full_tensor, test_predictions_full).numpy()[0][0]
    results[len(selected_features)] = full_metrics

    while len(selected_features) > 1:
        batch_combinations_idx = [[feature_names.index(f) for f in selected_features 
                                   if f != feature] for feature in selected_features]
        X_batches = [shap_train[:, combination] for combination in batch_combinations_idx]
        y_batches = [y_pred_train for _ in batch_combinations_idx]
        X_batches_tensor = tf.constant(np.stack(X_batches, axis=0), dtype=tf.float32)
        y_batches_tensor = tf.constant(np.stack(y_batches, axis=0), dtype=tf.float32)
        predictions_batches, weights_batches = _process_batch(X_batches_tensor, y_batches_tensor, n_samples)
        batch_r2_scores = _calculate_r2(y_batches_tensor, predictions_batches)

        best_combination_idx = np.argmax(batch_r2_scores.numpy())
        worst_feature = selected_features[best_combination_idx]
        remaining_features = [f for f in selected_features if f != worst_feature]

        iter_metrics = {'train_pred_r2': batch_r2_scores.numpy()[best_combination_idx][0]}

        # --- évaluation test : on réutilise les poids déjà fit sur train pour la meilleure combinaison ---
        if shap_test is not None and y_pred_test is not None:
            best_weights = weights_batches[best_combination_idx:best_combination_idx + 1]
            X_test_best = shap_test[:, [feature_names.index(f) for f in remaining_features]]
            X_test_tensor = tf.constant(X_test_best[np.newaxis, ...], dtype=tf.float32)
            y_test_tensor = tf.constant(y_pred_test[np.newaxis, ...], dtype=tf.float32)
            test_predictions = _predict_batch(X_test_tensor, best_weights, shap_test.shape[0])
            iter_metrics['test_pred_r2'] = _calculate_r2(y_test_tensor, test_predictions).numpy()[0][0]

        selected_features = remaining_features
        removed_features.append(worst_feature)
        results[len(selected_features)] = iter_metrics
        _progress_bar(len(removed_features) + 1, len(feature_names))

    removed_features.append(selected_features[0])
    removed_features.reverse()
    return removed_features, results


def shapNCK(
        shap_train: np.ndarray, y_pred_train: np.ndarray,
        feature_names: list,
        k: int = 2, 
        batch_size: int = 500, 
        selected_features: list = [],
        top: int = None
):
    """
    NCK stands for N choose K. This function uses Wrapshap to return the best combination of K features out of all N
    features. The compute needed for this function increases exponentially as k increases, be careful when going beyond 
    k=2 or k=3. 

    Parameters:
        shap_train (np.array): The SHAP values for the train set as a 2D array.
        y_pred_train (np.array): The complex model predictions for the train set as a 1D array.
        feature_names (list): The feature names.
        k (int): The is the K in N choose K.
        batch_size (int): The size of the batch of data to be processed simultenaously.
        selected_features (list): A list of already selected features that will be in the combination no matter what.
        top (int): When different than None, instead of performing NCK on all features, perfoms it only on the top 
                    features (the ones with biggest mean absolute SHAP values)

    Returns:
        tuple[list, list, list]: A tuple containing the best combination of K features as a list, the R2 scores of the
        exhaustive list of combinations that have been tested and the exhaustive list of combinations that have been
        tested. 
    """

    if top is not None:
        shap_train, feature_names = _select_top(top, shap_train, feature_names)

    y_pred_train = y_pred_train.reshape(-1,1)

    n_samples = shap_train.shape[0]
    r2_scores = []

    remaining_features = [feature for feature in feature_names if feature not in selected_features]
    
    feature_combinations = list(itertools.combinations(remaining_features, k))
    num_combinations = len(feature_combinations)

    print(f"Performing {num_combinations} linear regressions")

    for i in range(0, num_combinations, batch_size):
        batch_combinations = feature_combinations[i:i + batch_size]
        
        X_batch_data = np.array([shap_train[:, [feature_names.index(f) for f in selected_features + list(comb)]] 
                                 for comb in batch_combinations])
        
        y_batch_data = np.tile(y_pred_train, (len(batch_combinations), 1, 1))

        X_batch = tf.constant(X_batch_data, dtype=tf.float32)
        y_batch = tf.constant(y_batch_data, dtype=tf.float32)

        predictions_batch = _process_batch(X_batch, y_batch, n_samples)

        batch_r2_scores = _calculate_r2(y_batch, predictions_batch)
        r2_scores.extend(batch_r2_scores.numpy())

        print(f"Processed batch {i // batch_size + 1}/{1 + num_combinations // batch_size}")

    r2_scores = [r2[0] for r2 in r2_scores]
    max_r2_index = np.argmax(r2_scores)
    best_r2_score = r2_scores[max_r2_index]
    best_feature_combination = feature_combinations[max_r2_index]

    print("\nBest R2 Score:", best_r2_score)
    print("Best Feature Combination:", best_feature_combination)

    return best_feature_combination, r2_scores, feature_combinations


def shapXFFS(
        shap_train: np.ndarray, y_pred_train: np.ndarray,
        shap_test: np.ndarray, y_pred_test: np.ndarray,
        feature_names: list,
        n: int,
        max_depth: int,
        top: int = None,
        selected_features: list = None,
        current_depth: int = 0,
        unique_sequences: list = None,
        tree: dict = None
):
    """
    Explorative version of the Wrapshap GPU implementation of Forward Feature Selection (FFS). Instead of only selecting
    the best feature at each step, it selects the n best candidates and continues each n tree from there. 

    Parameters:
        shap_train (np.array): The SHAP values for the train set as a 2D array.
        y_pred_train (np.array): The complex model predictions for the train set as a 1D array.
        feature_names (list): The feature names.
        n (int): The number of candidates at each split (creating n branches)
        max_depth (int): The max depth of the explorative tree.
        top (int): When different than None, instead of performing XFFS on all features, perfoms it only on the top 
                    features (the ones with biggest mean absolute SHAP values)
    
    The following parameters are used by the recursive call of the function. Better not to touch them:
        selected_features (list)
        current_depth (int)
        unique_sequences (list)
        tree (dict)

    Returns:
        tuple[list, dict]: A tuple containing the list of ranked features (from most to least important) and a 
                        dictionary with the corresponding performance metrics.
    """

    if top is not None:
        shap_train, shap_test, feature_names = _select_top(top, shap_train, shap_test, feature_names)
    
    if selected_features is None:
        selected_features = []
    
    if tree is None:
        tree = {}
        
    if unique_sequences is None:
        unique_sequences = []

    if current_depth == max_depth or len(selected_features) >= len(feature_names):
        return tree
    n_samples = shap_train.shape[0]
    y_pred_train = y_pred_train.reshape(-1,1)
    if y_pred_test is not None:
        y_pred_test = y_pred_test.reshape(-1,1)
        
    feature_scores = []

    for feature in feature_names:
        if feature not in selected_features:
            current_features = selected_features + [feature]
            X_batch = shap_train[:, [feature_names.index(f) for f in current_features]]
            X_batch = X_batch.reshape(1, n_samples, -1) 
            y_batch = y_pred_train.reshape(1, n_samples, -1)
            
            X_batch_test = shap_test[:,[feature_names.index(f)for f in current_features]]
            X_batch_test = X_batch_test.reshape(1,shap_test.shape[0],-1)
            y_batch_test = y_pred_test.reshape(1,shap_test.shape[0],-1)
            
            predictions, weights = _process_batch(tf.constant(X_batch, dtype=tf.float32), tf.constant(y_batch, dtype=tf.float32), n_samples)
            test_predictions = _predict_batch(tf.constant(X_batch_test,dtype = tf.float32), weights, shap_test.shape[0])

            r2_score = _calculate_r2(y_batch, predictions).numpy()
            test_r2_score = _calculate_r2(y_batch_test, test_predictions).numpy()
            
            feature_scores.append((feature, r2_score, test_r2_score))

    best_features = sorted(feature_scores, key=lambda x: x[1], reverse=True)[:n]

    for best_feature, score, test_score in best_features:
        new_selected_features = selected_features + [best_feature]
        metrics = {'train_pred_r2' : score [0][0],
                   'test_pred_r2': test_score [0][0]}
        if sorted(new_selected_features) in [sorted(seq) for seq in unique_sequences]:
            if best_feature not in tree:
                tree[best_feature] = {'train_metrics': metrics, 'duplicate': 1}
            continue

        unique_sequences.append(new_selected_features)

        if best_feature not in tree:
            tree[best_feature] = {'train_metrics': metrics, 'duplicate': 0}
        else:
            tree[best_feature]['duplicate'] = 0

        if current_depth + 1 < max_depth and len(new_selected_features) < len(feature_names):
            tree[best_feature]['next_features'] = shapXFFS(
                shap_train=shap_train,
                shap_test=shap_test,
                y_pred_train=y_pred_train,
                y_pred_test=y_pred_test,
                feature_names=feature_names,
                n=n,
                max_depth=max_depth,
                selected_features=new_selected_features,
                current_depth=current_depth + 1,
                unique_sequences=unique_sequences
            )

    return tree


def print_XFFS_tree(
        tree: dict,
        metric: str = 'train_pred_r2',
        export_to_txt: bool = False,
        filename: str = 'tree.txt',
        hide_duplicates: bool = False
):
    """
    Utility function for a "pretty" print of the tree returned by shapXFFS.

    Parameters:
        tree (dict): The tree, the dictionary returned by shapXFFS.
        metric (str): The metric to print.
        export_to_txt (bool): Whether to export to a txt file or not.
        filename (str): The file name of the txt file.
        hide_duplicates: Whether to hide the duplicates or not.
    """

    def _get_all_branches(nested_dict, metric, depth=1, indent="", sequence_prefix="    ", is_sub_key=False):
        output_lines = []

        for key, value in nested_dict.items():
            if not hide_duplicates or value['duplicate']==0:
                current_prefix = f"{depth} --> " if value['duplicate']==0 else f"{depth} DUPLICATE --> "
                current_line = f"{indent}{current_prefix}{key} ({str(round(value['train_metrics'][metric], 2))})"
                output_lines.append(current_line)

                if 'next_features' in value:
                    deeper_lines = _get_all_branches(
                        value['next_features'],
                        metric, 
                        depth=depth + 1,
                        indent=indent + sequence_prefix,
                        sequence_prefix=sequence_prefix, 
                        is_sub_key=True
                    )
                    output_lines.extend(deeper_lines)

        return output_lines
    
    all_branches = _get_all_branches(tree, metric)

    all_branches_text = f"Displayed metric: {metric}\n\n" + '\n'.join(all_branches)
    print(all_branches_text)

    if export_to_txt:
        with open(filename, 'w') as file:
            file.write(all_branches_text)
            
            
def plot_XFFS_tree (tree: dict, 
                    metric: str ='train_pred_r2', 
                    save: bool = False, 
                    filename: str = 'tree.png'):

    
    color_map = plt.get_cmap('viridis')

    dot = Digraph(comment='Cluster 0 tree', graph_attr={'dpi': '300'})
    add_nodes_edges(dot ,tree, metric,parent_name=None, color_map=color_map, min_val=0, max_val=1)
    png_bytes = dot.pipe(format='png')
    tree_img = PILImage.open(io.BytesIO(png_bytes)).convert('RGB')
    
    # 2. Générer la colorbar seule, à la même hauteur que l'arbre
    fig, ax = plt.subplots(figsize=(0.5, tree_img.height / 300))  # hauteur ajustée au dpi
    norm = plt.Normalize(vmin=0, vmax=1)
    cb = fig.colorbar(cm.ScalarMappable(cmap='viridis', norm=norm), cax=ax)
    cb.set_label('valeur')  # à adapter selon votre légende
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    bar_img = PILImage.open(buf).convert('RGB')
    
    # 3. Redimensionner la colorbar pour qu'elle ait la même hauteur que l'arbre
    bar_img = bar_img.resize(
        (int(bar_img.width * tree_img.height / bar_img.height), tree_img.height)
    )
    
    # 4. Coller les deux images côte à côte
    combined = PILImage.new('RGB', (tree_img.width + bar_img.width, tree_img.height), 'white')
    combined.paste(tree_img, (0, 0))
    combined.paste(bar_img, (tree_img.width, 0))
    if save :
        output_path = filename
        dot.render(output_path, format='png', view=False)
    
    display(combined)
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



def _calculate_r2(y_true, y_pred):

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    y_pred = tf.reshape(y_pred, tf.shape(y_true))
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred), axis=1)
    y_true_mean = tf.reduce_mean(y_true, axis=1, keepdims=True)
    ss_tot = tf.reduce_sum(tf.square(y_true - y_true_mean), axis=1)
    r2_score = 1 - ss_res / ss_tot

    return r2_score


def _process_batch(X_batch, y_batch, n_samples):
    lambda_reg = 1
    X_batch = tf.cast(X_batch, tf.float64)
    y_batch = tf.cast(y_batch, tf.float64)
    ones = tf.ones((X_batch.shape[0], n_samples, 1), dtype=tf.float64)
    X_batch = tf.concat([ones, X_batch], axis=2)
    XTX = tf.matmul(X_batch, X_batch, transpose_a=True)
    lambda_identity = lambda_reg * tf.eye(XTX.shape[-1], batch_shape=[XTX.shape[0]], dtype=tf.float64)
    XTX_regularized = XTX + lambda_identity
    XTy = tf.matmul(X_batch, y_batch, transpose_a=True)
    weights = tf.linalg.solve(XTX_regularized, XTy)
    predictions = tf.matmul(X_batch, weights)
    predictions = tf.reshape(predictions, [X_batch.shape[0], n_samples])
    return tf.cast(predictions,tf.float32), tf.cast(weights, tf.float32)   # <-- on retourne aussi les poids


def _predict_batch(X_batch, weights, n_samples):
    """Applique des poids déjà appris (fit sur train) à de nouvelles données (ex: test)."""
    X_batch = tf.cast(X_batch, tf.float64)
    weights = tf.cast(weights, tf.float64)
    ones = tf.ones((X_batch.shape[0], n_samples, 1), dtype=tf.float64)
    X_batch = tf.concat([ones, X_batch], axis=2)
    predictions = tf.matmul(X_batch, weights)
    predictions = tf.reshape(predictions, [X_batch.shape[0], n_samples])

    return tf.cast(predictions,tf.float32)



### For the Tree Plot

def get_font_color(hex_color, threshold=0.5):
    # Convert hex to RGB tuple (values between 0 and 1)
    r, g, b = mcolors.to_rgb(hex_color)
    # Calculate luminance (a common approximation)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    # Return white for dark backgrounds, black for light backgrounds
    return 'white' if luminance < threshold else 'black'

    
# Function to add nodes and edges with emphasis on nodes where train_pred_r2 >= 0.90
def add_nodes_edges(graph, current_dict, metric, parent_name, color_map, min_val, max_val, threshold=0.90):
    for key, value in current_dict.items():
        node_name = str(uuid.uuid4())  # Generate a unique name for each node

        # Check for train_pred_r2 value and apply color mapping
        if 'train_metrics' in value:
            r2 = value['train_metrics'][metric]
            # Normalize the r2 value to get the color
            normalized_r2 = (r2 - min_val) / (max_val - min_val)
            color = color_map(normalized_r2)
            hex_color = mcolors.to_hex(color)
            # Determine the best font color based on the node's fill color
            fontcolor = get_font_color(hex_color)
            # Emphasize the node if r2 is above the threshold
            if r2 >= threshold:
                graph.node(node_name, label=f'{key}\n{r2:.2f}', 
                           fillcolor=hex_color, fontcolor=fontcolor, 
                           style='filled', shape='doubleoctagon', 
                           color='red', penwidth='3')
            else:
                graph.node(node_name, label=f'{key}\n{r2:.2f}',
                           fillcolor=hex_color, fontcolor=fontcolor, 
                           style='filled')
        else:
            graph.node(node_name, label=key)

        if parent_name:
            graph.edge(parent_name, node_name)

        # Recursively process next features
        if 'next_features' in value:
            add_nodes_edges(graph, value['next_features'],metric, node_name, color_map, min_val, max_val, threshold)

# Extract all r2 values from the nested dictionary for color mapping
def extract_r2_values(nested_dict, r2_values, metric):
    for key, value in nested_dict.items():
        if 'train_metrics' in value:
            r2_values.append(value['train_metrics'][metric])
        if 'next_features' in value:
            extract_r2_values(value['next_features'], r2_values, metric)
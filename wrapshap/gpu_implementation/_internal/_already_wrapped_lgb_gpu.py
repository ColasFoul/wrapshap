"""
This module provides utility functions and convenient classes of LightGBM regressors and classifiers already wrapped
with the GPU implementation of Wrapshap.
"""

# Imports 


import lightgbm as lgb

from ..wrapshap_gpu import WrapshapGPU
from ..._internal._already_wrapped_LGB import WrappedBinaryClassifierLGB, WrappedRegressorLGB, WrappedLGB
from ..._internal._already_wrapped import wrapshap_required

# ----------------------------------------------------------------------------------------------------------------------
# Public API - These utilities are used by other modules and re-exported.
# ----------------------------------------------------------------------------------------------------------------------


class WrappedBinaryClassifierLGB_GPU(WrappedBinaryClassifierLGB):

    def compute_shap_and_predictions(self):
        """
        Computes model's SHAP values and predictions and instantiates the underlying Wrapshap object.
        """

        WrappedLGB.compute_shap_and_predictions(self)

        self.y_pred_train = (
            self.model
            .predict(self.X_train, raw_score=True)
            .reshape(-1)
        )
        
        self.y_pred_test = (
            self.model
            .predict(self.X_test, raw_score=True)
            .reshape(-1)
        )
        self.wrapshap = WrapshapGPU(
            X_train=self.X_train, X_test=self.X_test, y_train=self.y_train, y_test=self.y_test,
            shap_train=self.shap_train, shap_test=self.shap_test, 
            y_pred_train=self.y_pred_train, y_pred_test=self.y_pred_test,
            feature_names=self.feature_names,
            task_type='classification'
        )

    @wrapshap_required('wrapshap')
    def shapFFS(self, surrogate_metrics: bool = False, top: int = None):
        """
        Simply calls Wrapshap GPU implementation of Forward Feature Selection (FFS). 

        Parameters:
            surrogate_metrics (bool): Whether the compute and return the surrogate metrics or not.
            top (int): When different than None, instead of performing FFS on all features, perfoms it only on the top 
                        features (the ones with biggest mean absolute SHAP values)

        Returns:
            tuple[list, dict]: A tuple containing the list of ranked features (from most to least important) and a 
                            dictionary with the corresponding performance metrics.
        """

        return self.wrapshap.shapFFS(surrogate_metrics=surrogate_metrics, top=top)
    

    @wrapshap_required('wrapshap')
    def shapBFE(self, surrogate_metrics: bool = False, top: int = None):
        """
        Simply calls Wrapshap implementation of Backward Feature Elimination (BFE). 

        Parameters:
            surrogate_metrics (bool): Whether the compute and return the surrogate metrics or not.
            top (int): When different than None, instead of performing BFE on all features, perfoms it only on the top 
                        features (the ones with biggest mean absolute SHAP values)

        Returns:
            tuple[list, dict]: A tuple containing the list of ranked features (from most to least important) and a 
                            dictionary with the corresponding performance metrics.
        """

        return self.wrapshap.shapBFE(surrogate_metrics=surrogate_metrics, top=top)
    
    @wrapshap_required('wrapshap')
    def shapNCK(self, k: int = 2, batch_size: int = 500, selected_features: list = [], top: int = None):
        """
        NCK stands for N choose K. This function uses Wrapshap to return the best combination of K features out of all N
        features. The compute needed for this function increases exponentially as k increases, be careful when going 
        beyond k=2 or k=3. 

        Parameters:
            k (int): The is the K in N choose K.
            batch_size (int): The size of the batch of data to be processed simultenaously.
            selected_features (list): A list of already selected features that will be in the combination no matter what.
            top (int): When different than None, instead of performing NCK on all features, perfoms it only on the top 
                        features (the ones with biggest mean absolute SHAP values)

        Returns:
            tuple[list, list, list]: A tuple containing the best combination of K features as a list, the R2 scores of 
            the exhaustive list of combinations that have been tested and the exhaustive list of combinations that have 
            been tested. 
        """
        return self.wrapshap.shapNCK(k=k, batch_size=batch_size, selected_features=selected_features, top=top)
    
    @wrapshap_required('wrapshap')
    def shapXFFS(self, n: int, max_depth: int, top: int = None):
        """
        Explorative version of the Wrapshap GPU implementation of Forward Feature Selection (FFS). Instead of only 
        selecting the best feature at each step, it selects the n best candidates and continues each n tree from there. 

        Parameters:
            n (int): The number of candidates at each split (creating n branches)
            max_depth (int): The max depth of the explorative tree.
            top (int): When different than None, instead of performing XFFS on all features, perfoms it only on the top 
                        features (the ones with biggest mean absolute SHAP values)

        Returns:
            tuple[list, dict]: A tuple containing the list of ranked features (from most to least important) and a 
                            dictionary with the corresponding performance metrics.
        """
        return self.wrapshap.shapXFFS(n=n, max_depth=max_depth, top=top)
    


class WrappedRegressorLGB_GPU(WrappedRegressorLGB):

    def compute_shap_and_predictions(self):
        """
        Computes model's SHAP values and predictions and instantiates the underlying Wrapshap object.
        """

        WrappedLGB.compute_shap_and_predictions(self)

        self.y_pred_train = self.model.predict(self.X_train).reshape(-1)
        self.y_pred_test = self.model.predict(self.X_test).reshape(-1)
        
        self.wrapshap = WrapshapGPU(
            X_train=self.X_train, X_test=self.X_test, y_train=self.y_train, y_test=self.y_test,
            shap_train=self.shap_train, shap_test=self.shap_test, 
            y_pred_train=self.y_pred_train, y_pred_test=self.y_pred_test,
            feature_names=self.feature_names,
            task_type='regression'
        )

    @wrapshap_required('wrapshap')
    def shapFFS(self, surrogate_metrics: bool = False, top: int = None):
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

        return self.wrapshap.shapFFS(surrogate_metrics=surrogate_metrics, top=top)
    

    @wrapshap_required('wrapshap')
    def shapBFE(self, surrogate_metrics: bool = False, top: int = None):
        """
        Simply calls Wrapshap implementation of Backward Feature Elimination (BFE). 

        Parameters:
            surrogate_metrics (bool): Whether the compute and return the surrogate metrics or not.
            top (int): When different than None, instead of performing BFE on all features, perfoms it only on the top 
                        features (the ones with biggest mean absolute SHAP values)

        Returns:
            tuple[list, dict]: A tuple containing the list of ranked features (from most to least important) and a 
                            dictionary with the corresponding performance metrics.
        """

        return self.wrapshap.shapBFE(surrogate_metrics=surrogate_metrics, top=top)
    
    
    @wrapshap_required('wrapshap')
    def shapNCK(self, k: int = 2, batch_size: int = 500, selected_features: list = [], top: int = None):
        """
        NCK stands for N choose K. This function uses Wrapshap to return the best combination of K features out of all N
        features. The compute needed for this function increases exponentially as k increases, be careful when going 
        beyond k=2 or k=3. 

        Parameters:
            k (int): The is the K in N choose K.
            batch_size (int): The size of the batch of data to be processed simultenaously.
            selected_features (list): A list of already selected features that will be in the combination no matter what.
            top (int): When different than None, instead of performing NCK on all features, perfoms it only on the top 
                        features (the ones with biggest mean absolute SHAP values)

        Returns:
            tuple[list, list, list]: A tuple containing the best combination of K features as a list, the R2 scores of 
            the exhaustive list of combinations that have been tested and the exhaustive list of combinations that have 
            been tested. 
        """
        return self.wrapshap.shapNCK(k=k, batch_size=batch_size, selected_features=selected_features, top=top)
    
    @wrapshap_required('wrapshap')
    def shapXFFS(self, n: int, max_depth: int, top: int = None):
        """
        Explorative version of the Wrapshap GPU implementation of Forward Feature Selection (FFS). Instead of only 
        selecting the best feature at each step, it selects the n best candidates and continues each n trees from there. 

        Parameters:
            n (int): The number of candidates at each split (creating n branches).
            max_depth (int): The max depth of the explorative tree.
            top (int): When different than None, instead of performing XFFS on all features, perfoms it only on the top 
                        features (the ones with biggest mean absolute SHAP values)

        Returns:
            tuple[list, dict]: A tuple containing the list of ranked features (from most to least important) and a 
                            dictionary with the corresponding performance metrics.
        """
        return self.wrapshap.shapXFFS(n=n, max_depth=max_depth, top=top)

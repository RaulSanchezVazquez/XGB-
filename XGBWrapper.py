#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 16:06:52 2017
@author: raulsanchez
"""
import warnings; warnings.filterwarnings('ignore');

import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class XGBWrapper():
    def __init__(
        self,
        **kwards):
        """
        XGB wrapper to make it compatible with SKlearn GridSearch
        """
        self.xgb_classifier = None
        
        self.init_params = {
            #Maximum tree depth for base learners.    
            'max_depth' : kwards.pop('max_depth', 3),
            #Boosting learning rate (xgb's "eta")
            'learning_rate' : kwards.pop('learning_rate', .1),
            #Number of boosted trees to fit.
            'n_estimators' : kwards.pop('n_estimators', 100),
            #Whether to print messages while running boosting.
            'silent' : kwards.pop('silent', False),
            #Specify the learning task and the corresponding learning objective 
            #or a custom objective function to be used (see note below).
            'objective' : kwards.pop('objective', 'binary:logistic'),
            #Specify which booster to use: gbtree, gblinear or dart.
            'booster' : kwards.pop('booster', 'gbtree'),
            # Number of parallel threads used to run xgboost.
            'n_jobs' : kwards.pop('n_jobs', 1),
            # Minimum loss reduction required to make a further partition on a 
            #leaf node of the tree.
            'gamma' : kwards.pop('gamma', 0),
            # Minimum sum of instance weight(hessian) needed in a child.
            'min_child_weight' : kwards.pop('min_child_weight', 1),
            # Maximum delta step we allow each tree's weight estimation to be.
            'max_delta_step' : kwards.pop('max_delta_step', 0),
            # Subsample ratio of the training instance.
            'subsample' : kwards.pop('subsample', 1),
            # Subsample ratio of columns when constructing each tree.
            'colsample_bytree' : kwards.pop('colsample_bytree', 1),
            # Subsample ratio of columns for each split, in each level.
            'colsample_bylevel' : kwards.pop('colsample_bylevel', 1),
            # L1 regularization term on weights
            'reg_alpha' : kwards.pop('reg_alpha', 0),
            # L2 regularization term on weights
            'reg_lambda' : kwards.pop('reg_lambda', 1),
            # Balancing of positive and negative weights.
            'scale_pos_weight' : kwards.pop('scale_pos_weight', 1),
            # The initial prediction score of all instances, global bias.
            'base_scor' : kwards.pop('base_scor', .5),
            # Random number seed.  (Deprecated, please use random_state)
            'seed' : kwards.pop('seed', None),
            # Random number seed.  (replaces seed)
            'random_state' : kwards.pop('random_state', 0),
            # Value in the data which needs to be present as a missing value. 
            #If None, defaults to np.nan.
            'missing' : kwards.pop('missing', None)
        }
        
        self.fit_params = {
            #(array_like) – Weight for each instance
            'sample_weight' : kwards.pop('sample_weight', None),
            #(list, optional) – A list of (X, y) pairs to use as a validation 
            #set for early-stopping
            'eval_set': kwards.pop('eval_set', None),
            #(str, callable, optional) – If a str, should be a built-in 
            #evaluation metric to use.
            'eval_metric': kwards.pop('eval_metric', None),
            #(int, optional) – Activates early stopping. Validation error needs 
            #to decrease at least every <early_stopping_rounds> round(s) 
            #to continue training.
            'early_stopping_rounds': kwards.pop('early_stopping_rounds', None),
            #bool) – If verbose and an evaluation set is used, writes the 
            #evaluation metric measured on the validation set to stderr.
            'verbose': kwards.pop('verbose', False),
            #str) – file name of stored xgb model or ‘Booster’ instance 
            #Xgb model to be loaded before training (allows training 
            #continuation).
            'xgb_model': kwards.pop('xgb_model', None),
        }
        
        self.set_alg()
        
    def set_alg(self):
        """ 
        Instanciates the xgb model with init_parameters
        """
        
        self.xgb_classifier = xgb.XGBClassifier(
            max_depth=self.init_params['max_depth'],
            learning_rate=self.init_params['learning_rate'],
            n_estimators=self.init_params['n_estimators'],
            silent=self.init_params['silent'],
            objective=self.init_params['objective'],
            booster=self.init_params['booster'],
            n_jobs=self.init_params['n_jobs'],
            gamma=self.init_params['gamma'],
            min_child_weight=self.init_params['min_child_weight'],
            max_delta_step=self.init_params['max_delta_step'],
            subsample=self.init_params['subsample'],
            colsample_bytree=self.init_params['colsample_bytree'],
            colsample_bylevel=self.init_params['colsample_bylevel'],
            reg_alpha=self.init_params['reg_alpha'],
            reg_lambda=self.init_params['reg_lambda'],
            scale_pos_weight=self.init_params['scale_pos_weight'],
            base_scor=self.init_params['base_scor'],
            seed=self.init_params['seed'],
            random_state=self.init_params['random_state'],
            missing=self.init_params['missing'])
        
    def fit(self, X, y):
        """
        Fits the xgb model and sets fit_params
        """
        
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = pd.DataFrame(X)
        self.y_ = pd.Series(y)

        self.xgb_classifier.fit(
            self.X_, self.y_,
            early_stopping_rounds=(
                    self.fit_params['early_stopping_rounds']),
            eval_metric=self.fit_params['eval_metric'],
            sample_weight=self.fit_params['sample_weight'],
            eval_set=self.fit_params['eval_set'],
            verbose=self.fit_params['verbose'])
            
        return self
    
    def predict(self, X):
        """ XGB prediction """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)
        X = pd.DataFrame(X)
        
        return self.xgb_classifier.predict(X)

    def predict_proba(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)
        X = pd.DataFrame(X)
        
        """ XGB predict_proba"""
        return self.xgb_classifier.predict_proba(X)
    
    def get_params(self, deep=False):
        """ get_params for sklearn interface """
        return self.xgb_classifier.get_params()
    
    def set_params(self, **params):
        """ set_params for sklearn interface """
        for param_name, param_val in params.items():
            if param_name in self.fit_params:
                self.fit_params[param_name] = param_val
            elif param_name in self.init_params:
                self.init_params[param_name] = param_val
            else:
                raise ValueError('Invalid param: %s' % param_name)
        self.set_alg()
        return self

def test():
    
    from sklearn.model_selection import GridSearchCV
    from evalutils import class_report
    from evalutils import testdata
    
    X_train, X_test, y_train, y_test = testdata()
    # GridSearch
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [.0001, .001, .01],
        'n_estimators': [50, 75, 100, 150],
        'eval_set':[[(X_test, y_test)]],
        'n_jobs':[4],
        'early_stopping_rounds': [5, 10, 30]
    }
    
    #//XGB///////////////////////////////////
    
    gs = GridSearchCV(
            XGBWrapper(),
            param_grid=param_grid,
            scoring='accuracy',
            cv=2)
    
    gs.fit(X_train, y_train)
    
    #//EVAL///////////////////////////////////
    y_hat = gs.predict(X_test)
    
    report_bin = class_report(y_true=y_test, y_pred=y_hat)
    print(report_bin)
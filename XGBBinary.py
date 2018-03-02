#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 16:06:52 2017
@author: raulsanchez
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

import time
import os

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import classification_report

def pd_classification_report(
        y_true, 
        y_pred, 
        target_names=None, 
        sample_weight=None, 
        digits=2):
    """
    Turns the sklearn.metric.classification_report
    to a pandas.DataFrame
    """

    y_true = pd.Series(y_true).astype(str)
    y_pred = pd.Series(y_pred).astype(str)

    y_true = y_true.str.replace(' ', '_')
    y_pred = y_pred.str.replace(' ', '_')

    report = classification_report(
        y_true, 
        y_pred,
        target_names=target_names, 
        sample_weight=sample_weight,
        digits=digits)

    header = []
    for x in report.split("\n")[0].split(' '):
        if x != '':
            header.append(x)

    report = report.replace('avg / total ', 'all')
    lines = report.split("\n")
    all_info = []
    for line in lines[2:-1]:
        if line == '':
            continue

        info = []
        for x in line.split(' '):
            if x !='':
                info.append(x)
        all_info.append(info)

    report_df = pd.DataFrame(all_info, columns=['y']+header)

    class_names = report_df['y'].tolist()
    class_names[-1] = 'Global'
    report_df.index = class_names

    vc_pred = pd.Series(y_pred).value_counts()

    report_df['pred'] = vc_pred

    report_df.loc['Global', 'pred'] = report_df['pred'].sum()

    for c in report_df.columns[1:]:
        report_df[c] = report_df[c].astype(float)

    return report_df

def xgb_importances(xgb_object, class_label=''):
    """ 
    XGB
    
    Prints the three xgb metrics for feature importance:
        -'weight' the number of times a feature is used to split the data 
            across all trees.
        - 'gain' the average gain of the feature when it is used in trees.
        - 'cover' the average coverage of the feature when it is used in trees.
    
    """
    booster = xgb_object.alg.get_booster()
    
    imp_scores = pd.DataFrame()
    for importance_type in ['weight', 'gain', 'cover']:
        scores = pd.Series(
            booster.get_score(importance_type=importance_type))
        imp_scores[importance_type] = scores
    
    new_index = []
    for x in imp_scores.index:
        x = x.replace("$", " ")
        new_index.append(x)
    imp_scores.index = new_index
        
    imp_scores_norm = imp_scores / imp_scores.sum(axis=0)
    
    imp_scores_norm['mixture'] = imp_scores_norm.sum(axis=1)
    
    imp_scores_norm.sort_values(
        ['mixture', 'gain', 'cover'], 
        inplace=True)
    if imp_scores_norm.shape[0] > 20:
        imp_scores_norm = imp_scores_norm.iloc[-20:]
        
    fig, ax = plt.subplots(1, 1)
    imp_scores_norm.drop(['mixture'], axis=1).plot(
        kind='bar', figsize=(12, 12), 
        ax=ax,
        fontsize=9,
        title='Class_%s' % class_label)        
    
    fig.set_tight_layout('tight')
    
    return fig, ax

def plot_importance_binmod(
    model,
    label='target_30_60_90',
    folder_path='/home/raulsanchez/inter-prj-imobiliario/docs/img/'):
    """
    
    """
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    
    for class_id, xgb_object in model.ens_models.items():
        fig, ax = xgb_importances(xgb_object)
        
        filename = ("%s_class_%s.eps" % (label, class_id))
        fig.savefig(
            folder_path + filename, 
            bbox_inches='tight',
            type='eps')

class XGBWrapper():
    def __init__(
        self,
        **kwards):
        """
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
            'silent' : kwards.pop('silent', True),
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
            'verbose': kwards.pop('verbose', True),
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
        
        self.xgb_classifier.fit(
            X, y,
            early_stopping_rounds=(
                    self.fit_params['early_stopping_rounds']),
            eval_metric=self.fit_params['eval_metric'],
            sample_weight=self.fit_params['sample_weight'],
            eval_set=self.fit_params['eval_set'],
            verbose=self.fit_params['verbose'])
            
        return self
    
    def predict(self, X):
        """ XGB prediction """
        return self.xgb_classifier.predict(X)

    def predict_proba(self, X):
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

class XGBBinary(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        gridsearch,
        gridsearch_n_jobs=4,
        gridsearch_verbose=True,
        performance_politic='recall__min',
        cross_validation=3,
        equal_sample=False,
        test=.5,
        verbose=True):
        
        """ Training params """
        self.test = test
        self.equal_sample = equal_sample
        self.cross_validation = cross_validation
        
        """ GridSearch params """
        self.gridsearch = gridsearch
        self.gridsearch_n_jobs = gridsearch_n_jobs,
        self.gridsearch_verbose = gridsearch_verbose
        
        """ Performance politic """
        self.performance_politic = performance_politic
        
        self.verbose = verbose
        
        """ Variables to hold data"""
        self.n = None
        self.d = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        """ Debugging Variables """
        self.logs = ""
        self.train_report = None
        self.test_report = None
        
    def init_train(self, X, y):
        """
        Stores the data splits and records basic statistics
        """
        
        self.X_raw, self.y_raw = (
                pd.DataFrame(X), 
                pd.Series(y))
        
        """ Splits train and test """
        if self.test > 0:
            (self.X_train,
             self.X_test,
             self.y_train,
             self.y_test) = train_test_split(
                self.X_raw, self.y_raw, test_size=self.test)
        else:
            self.X_train = self.X_raw
            self.y_train = self.y_raw
        
        """ Number of instances """
        self.n = self.X_train.shape[0]

        """ Data dimensions """
        self.d = self.X_train.shape[1]
        
        """ Number of classes """
        self.class_labels = np.unique(self.y_raw)
        self.n_classes = len(self.class_labels)
        
        self.ens_models = dict([(l, None) for l in self.class_labels])

    def fit(self, X, y):
        """
        Fits the model
        """
        
        """ Stores the data splits and records basic statistics """
        self.init_train(X, y)
        
        """ Trains a Binary Classifier with grid-search & cross-validation """
        self.train()
        
        

    def train(self):
        """
        Trains a Binary Classifier with grid-search and cross-validation
        """
        
        for class_label in self.class_labels:
            #class_label = 0
            gridsearch = self.gridsearch.copy()
            msg = "Classifier label %s" % (class_label)
            self.logs += msg+"\n" 
            if self.verbose:
                print(msg)
                
            X_local = self.X_train
            
            y_local = self.get_target_binary(self.y_train, class_label)
            y_local_test = self.get_target_binary(self.y_test, class_label)
            
            num_pos_class = (y_local == 1).sum()
            num_neg_class = (y_local == 0).sum()
            
            scale_pos_weight = num_neg_class / num_pos_class
            
            gridsearch['scale_pos_weight'] = [scale_pos_weight]
            
            model = self.binary_GridSearchCV(
                X_local, y_local, y_local_test, gridsearch)
            
            """ Add to Ens. """
            self.ens_models[class_label] = model

        """ Evaluate Model """
        self.eval()
        
    def find_best_params_GridSearchCV(self, gridsearch, X_local, y_local):
        """
        Find best set of parameters given a grid-search round 
        across various folds.
        """
        
        """ Generates all combination of hyperparameters """
        hyperparam_space = list(ParameterGrid(gridsearch))
        
        if len(hyperparam_space) == 1:
            return hyperparam_space[0]
        
        """ Splits local folds """
        kf = KFold(n_splits=self.cross_validation)
        kf.get_n_splits(self.X_train)
        
        folds = []
        for train_index, test_index in kf.split(X_local):
            folds.append([
            pd.Series(train_index), 
            pd.Series(test_index)])
        
        if self.verbose:
            n_folds = len(folds) 
            n_combinations = len(hyperparam_space)
            
            msg = "%s models to fit (%s folds x %s param. combinations) ...\n"
            print(msg % (
                    n_folds * n_combinations,
                    n_folds, 
                    n_combinations))
        
        """ Performs gridsearch """
        #Stores performance, Stores classification reports
        performance = []
        for params_it, params in enumerate(hyperparam_space):
            time_start = time.time()
            
            #Evaluation rounds
            local_results = []
            for fold_it, fold in enumerate(folds):
                X_train = X_local.iloc[fold[0]]
                X_test  = X_local.iloc[fold[1]]
                y_train = y_local.iloc[fold[0]]
                y_test = y_local.iloc[fold[1]]
                
                params['eval_set'] = [(X_test, y_test)]
                
                alg = XGBWrapper(**params)
                alg.fit(X_train, y_train)
                
                pred_test = alg.predict(X_test)
                
                local_report = report_to_df(
                    y_true=y_test, 
                    y_pred=pred_test)
                
                local_results.append(local_report)
            
            #Stores performance evaluation given the performance-policy
            
            #self.performance_politic = 'recall-min'
            
            metric, statistic = self.performance_politic.split('__')
            local_performance = []
            
            for local_report in local_results:
                local_report = local_report.drop('Global')
                metric_results = local_report[metric]
                
                if statistic == 'min':
                    metric_stat = metric_results.min()
                elif statistic == 'max':
                    metric_stat = metric_results.max()
                elif statistic == 'mean':
                    metric_stat = metric_results.mean()
                
                local_performance.append(metric_stat)
            
            local_performance = pd.Series(local_performance)
            performance.append(local_performance)
            
            time_end = time.time()
            elapsed_time = (time_end - time_start)
            
            if self.gridsearch_verbose:
                msg = "%s of %s - %s: %s  - %s s" % (
                    (params_it + 1), 
                    n_combinations,
                    self.performance_politic,
                    round(local_performance.mean(), 4),
                    round(elapsed_time, 2))
                
                print(msg)
                
                for param_name in params.keys():
                    if param_name != 'eval_set':
                        msg = "\t%s: %r" % (param_name, params[param_name])
                        if self.verbose:
                            print(msg)
                print('')
                
        performance = pd.DataFrame(performance)
        
        mean_performance = performance.mean(axis=1)
        idx_best = mean_performance.idxmax()
        best_parameters = hyperparam_space[idx_best]
        
        return best_parameters

    def binary_GridSearchCV(self, X_local, y_local, y_local_test, gridsearch):
        """
        Trains an algorithm with grid-search and cross-validation
        expects two classes
        """
        
        best_parameters = self.find_best_params_GridSearchCV(
            gridsearch=gridsearch,
            X_local=X_local,
            y_local=y_local)
        
        #print best parameters
        print("\\\\\\\\Winner model:")
        for param_name in sorted(self.gridsearch.keys()):
            if param_name in best_parameters:
                msg = "\t%s: %r" % (param_name, best_parameters[param_name])
                self.logs += msg+"\n" 
                
                if self.verbose:
                    print(msg)
        
        #fit new model after parameters are fixed
        best_parameters['eval_set'] = [(self.X_test, y_local_test)]
        
        """
        xgb_model = xgb.XGBClassifier(
            learning_rate=0.1,
            max_depth=5,
            scale_pos_weight=9.5421686746987948)
        
        xgb_model.fit(
                self.X_train, y_local, 
                early_stopping_rounds=best_parameters['early_stopping'],
                eval_set=best_parameters['eval_set'])
        
        y_local.value_counts()
        
        xgb_yhat = xgb_model.predict(self.X_train)
        
        pd.Series(xgb_yhat).value_counts()
        
        report_to_df(y_true=y_local, y_pred=xgb_yhat)
        """
        
        best_model = XGBWrapper(**best_parameters)
        best_model.fit(self.X_train, y_local)
        
        """
        local_pred_train = best_model.predict(self.X_train)
        report_to_df(
            y_true=y_local, 
            y_pred=local_pred_train)
        """
        
        local_pred = best_model.predict(self.X_test)
        local_report = report_to_df(
            y_true=y_local_test, 
            y_pred=local_pred)
        
        msg = str(local_report) + "\n-----------------------------"
        self.logs += msg + '\n'
        print(msg)
        
        return best_model

    def train_individual(self, y_local, y_local_test, scale_pos_weight):
        """ Fit the model """
        model = self.alg(scale_pos_weight=scale_pos_weight, **self.alg_params)
        
        model.fit(self.X_train, y_local)
        
        return model
    
    def get_target_binary(self, y, class_label):
        return (y == class_label).astype(int)
    
    def eval(self):
        """
        Evals current performance relative to Train/Test
        """

        """ Train Ens. Predictions """
        y_train_hat = self.predict(self.X_train)
        
        if self.test > 0:
            """ Test Ens. Predictions """
            y_test_hat = self.predict(self.X_test)

        """ F1. Score """
        train_report = report_to_df(
            y_true=self.y_train, 
            y_pred=y_train_hat)
        self.train_report = train_report 
        
        if self.test > 0:
            test_report = report_to_df(
                y_true=self.y_test, 
                y_pred=y_test_hat)
            self.test_report = test_report
        
        msg = "////////Global Train////////"
        self.logs += msg + "\n" + str(train_report) + "\n"
        
        if self.verbose:
            print(msg)
            print(train_report)
            
            msg = "////////Global Test////////"
            self.logs += msg + "\n" + str(test_report) + "\n"
            if self.test > 0:
                print(msg)
                print(test_report)

    def predict(self, X, details=False):
        """
        Ensemble Predictions
        """
        X = pd.DataFrame(X)
        ens_pred = pd.DataFrame()
        for class_label in self.class_labels:
            y_pred_local = self.ens_models[class_label].predict_proba(X)
            ens_pred[class_label] = pd.DataFrame(y_pred_local)[1]
        
        if details:
            return ens_pred
        else:
            y_pred = np.argmax(ens_pred.values, axis=1)
            return y_pred
    
    def load(self, filename):
        """
        Loads a pickle object
        """

        f = open(filename, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        
        self.__dict__.update(tmp_dict)

    def save(self, filename):
        """
        Saves as pickle object
        """

        f = open(filename, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()
    
    def print_logs(self):
        """ Print Logs """
        print(self.logs)

def test_wrapper():
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
            n_samples=10000, 
            n_features=20, 
            n_informative=6,
            flip_y=10,
            n_redundant=2,
            n_repeated=0, 
            n_classes=10, 
            n_clusters_per_class=2,
            random_state=1)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    
    split_th = 7000
    X_train, X_test, y_train, y_test = (
            X.iloc[:split_th ],
            X.iloc[split_th:],
            y.iloc[:split_th ],
            y.iloc[split_th:])
    
    #/////////////////////////////////////////////////////
    xgb_wrapper = XGBWrapper(
            learning_rate=.1, 
            seed=1, 
            eval_set=[(X_test, y_test)])
    
    xgb_normal = xgb.XGBClassifier(
            learning_rate=.1, 
            seed=1)
    
    xgb_wrapper.fit(X_train, y_train)
    xgb_normal.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    
    print("Test Wrapper vs XGB")
    #self=model
    y_pred_test_wrapper = xgb_wrapper.predict(X_test)
    report_wrapper = report_to_df(
            y_true=y_test, 
            y_pred=y_pred_test_wrapper)
    print(report_wrapper)
    
    y_pred_test_normal = xgb_normal.predict(X_test)
    report_normal = report_to_df(
            y_true=y_test, 
            y_pred=y_pred_test_normal)
    print(report_normal)
    
    diff_results = (report_wrapper != report_normal).sum().sum()
    print(diff_results)
    
def test():
    n_classes = 5
    X, y = make_classification(
            n_samples=200000,
            #n_features=20,
            n_informative=6,
            #flip_y=1,
            #n_redundant=2,
            #n_repeated=0, 
            n_classes=n_classes, 
            weights=np.array([.5/(n_classes-1)]*(n_classes-1)).tolist() + [.5],
            #n_clusters_per_class=2,
            #random_state=1
            )
    
    X = pd.DataFrame(X)
    y = pd.Series(y)
    
    split_th = int(X.shape[0]/2)
    X_train, X_test, y_train, y_test = (
            X.iloc[:split_th ], 
            X.iloc[split_th:], 
            y.iloc[:split_th ], 
            y.iloc[split_th:])
    y_train.value_counts()
    #//BINARY///////////////////////////////////
    gridsearch = {
        'max_depth': [3, 5, 10],
        'learning_rate': [.10, .01, .001],
        'early_stopping_rounds': [5, 10, 15],
        'n_jobs':[4],
    }

    model = XGBBinary(
        gridsearch=gridsearch,
        gridsearch_n_jobs=4,
        gridsearch_verbose=True,
        test=0.3,
        cross_validation=2,
        performance_politic='f1-score__max',
        equal_sample=False)
    
    #self = model
    model.fit(X_train, y_train)
    
    #//XGB///////////////////////////////////
    xgb_model = xgb.XGBClassifier(n_jobs=4)
    xgb_model.fit(X_train, y_train)
    
    #//EVAL///////////////////////////////////
    print("-------Test vs XGB-------")
    print("-------Binary XGB-------")
    #self=model
    y_hat = model.predict(X_test)
    report_bin = report_to_df(y_true=y_test, y_pred=y_hat)
    print(report_bin)
    
    print("-------XGB-------")
    
    xgb_yhat = xgb_model.predict(X_test)
    report_xgb = report_to_df(y_true=y_test, y_pred=xgb_yhat)
    print(report_xgb)
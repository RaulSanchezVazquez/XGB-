#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 16:06:52 2017
@author: raulsanchez
"""
import warnings; warnings.filterwarnings('ignore');

import pickle
import pandas as pd
import numpy as np
import xgboost as xgb

import time

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid
from evalutils import class_report
import XGBWrapper as xgbw

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
            
            y_local = self.get_target_binary(
                self.y_train, 
                class_label)
            y_local_test = self.get_target_binary(
                self.y_test, 
                class_label)
            
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
            
            msg = "Fitting %s models...\n"
            print(msg % (n_combinations))
        
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
                
                alg = xgbw.XGBWrapper(**params)
                alg.fit(X_train, y_train)
                
                pred_test = alg.predict(X_test)
                
                local_report = class_report(
                    y_true=y_test, 
                    y_pred=pred_test)
                
                local_results.append(local_report)
            
            #Stores performance evaluation given the performance-policy
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
                        msg = "\t%s: %r" % (
                            param_name, 
                            params[param_name])
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
        print("----Best model---")
        for param_name in sorted(self.gridsearch.keys()):
            if param_name in best_parameters:
                msg = "%s: %r" % (
                    param_name, 
                    best_parameters[param_name])
                self.logs += msg+"\n" 
                
                if self.verbose:
                    print(msg)
        print("-----------------")
        #fit new model after parameters are fixed
        best_parameters['eval_set'] = [(self.X_test, y_local_test)]
        
        best_model = xgbw.XGBWrapper(**best_parameters)
        best_model.fit(self.X_train, y_local)
        
        local_pred = best_model.predict(self.X_test)
        local_report = class_report(
            y_true=y_local_test, 
            y_pred=local_pred)
        
        msg = str(local_report) + "\n-----------------------------"
        self.logs += msg + '\n'
        print(msg)
        
        return best_model

    def train_individual(self, y_local, y_local_test, scale_pos_weight):
        """ Fit the model """
        model = self.alg(
            scale_pos_weight=scale_pos_weight, 
            **self.alg_params)
        
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
        train_report = class_report(
            y_true=self.y_train, 
            y_pred=y_train_hat)
        self.train_report = train_report 
        
        if self.test > 0:
            test_report = class_report(
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
    
    def predict_proba(self, X):
        """
        Sklearn model.predict_proba()
        """
        
        return self.predict(X, details=True)
                            
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
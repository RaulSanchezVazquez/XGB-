#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 18:50:41 2018

@author: raulsanchez
"""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification

def testdata(
    n_classes=5, 
    n_samples=100000, 
    super_class_percent=.7):
    """
    Make Test DataSet
    """

    n_clusters_per_class = 2

    n_informative = n_classes * n_clusters_per_class
    n_features = int(n_informative * 1.5)

    other_class_percent = (
        (1 - super_class_percent) / (n_classes-1))

    weights = [other_class_percent]*(n_classes-1)
    weights.append(super_class_percent)

    X, y = make_classification(
            n_samples=n_samples,
            n_informative=n_informative,
            n_classes=n_classes,
            n_features=n_features,
            weights=weights)

    X = pd.DataFrame(X)
    y = pd.Series(y)

    split_th = int(X.shape[0]/2)
    X_train, X_test, y_train, y_test = (
            X.iloc[:split_th ], 
            X.iloc[split_th:], 
            y.iloc[:split_th ], 
            y.iloc[split_th:])

    return X_train, X_test, y_train, y_test
    
def class_report(
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
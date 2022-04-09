#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
import pandas as pd
from loglizer.models import *
from loglizer import dataloader, preprocessing

from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support

# run_models = ['PCA', 'InvariantsMiner', 'LogClustering', 'IsolationForest', 'LR',
#               'SVM', 'DecisionTree']

run_models = ['SVM']

struct_log = '../data/HDFS/HDFS.npz'# The benchmark dataset

if __name__ == '__main__':
    (x_tr, y_train), (x_te, y_test) = dataloader.load_HDFS(struct_log,
                                                           window='session',
                                                           train_ratio=0.5,
                                                           split_type='uniform')
    benchmark_results = []
    print('run_models', run_models)
    for _model in run_models:
        print('Evaluating {} on HDFS:'.format(_model))
        if _model == 'PCA':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf',
                                                      normalization='zero-mean')
            model = PCA()
            model.fit(x_train)

        elif _model == 'InvariantsMiner':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr)
            model = InvariantsMiner(epsilon=0.5)
            model.fit(x_train)

        elif _model == 'LogClustering':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf')
            model = LogClustering(max_dist=0.3, anomaly_threshold=0.3)
            model.fit(x_train[y_train == 0, :]) # Use only normal samples for training

        elif _model == 'IsolationForest':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr)
            model = IsolationForest(random_state=2019, max_samples=0.9999, contamination=0.03,
                                    n_jobs=4)
            model.fit(x_train)

        elif _model == 'LR':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf')
            model = LR()
            model.fit(x_train, y_train)

        elif _model == 'SVM':
            feature_extractor = preprocessing.FeatureExtractor()

            print('x_train', x_tr)

            x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf')

            print('x_train', x_train)

            penalty = 'l1'
            tol = 0.1
            C = 1
            dual = False
            class_weight = None
            max_iter = 100

            model = svm.LinearSVC(penalty=penalty, tol=tol, C=C, dual=dual,
                            class_weight=class_weight, max_iter=max_iter)
            # model = SVM()
            model.fit(x_train, y_train)
            y_train_predict = model.predict(x_train)

        elif _model == 'DecisionTree':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf')
            model = DecisionTree()
            model.fit(x_train, y_train)
        
        x_test = feature_extractor.transform(x_te)

        print('Train accuracy:')
        precision, recall, f1, _ = precision_recall_fscore_support(y_train, y_train_predict, average='binary')
        # precision, recall, f1 = model.evaluate(x_train, y_train)
        benchmark_results.append([_model + '-train', precision, recall, f1])
        print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))

        print('Test accuracy:')
        y_pred = model.predict(x_test)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        # precision, recall, f1 = model.evaluate(x_test, y_test)
        benchmark_results.append([_model + '-test', precision, recall, f1])
        print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))

    pd.DataFrame(benchmark_results, columns=['Model', 'Precision', 'Recall', 'F1']) \
      .to_csv('benchmark_result.csv', index=False)

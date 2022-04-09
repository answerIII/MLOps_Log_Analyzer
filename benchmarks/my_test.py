import os
import sys
import numpy as np

import my_dataloader

from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM
from sklearn.kernel_approximation import Nystroem
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import xgboost as xgb
from catboost import CatBoostClassifier

import tensorflow as tf
import tensorflow_hub as hub
from collections import OrderedDict

import my_tf_idf

from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append('../')
SEED = 42


def join_messages(X):
    x_trans = ['. '.join(x) for x in X]
    return x_trans


def models_cheker(x_train, y_train, x_test, y_test, models):
    models_results = {}
    for model_name in models:
        if model_name == 'CatBoost':
            print('CatBoost model')
            model = CatBoostClassifier(iterations=2, learning_rate=1, depth=2)
            x_train = np.array(x_train)
            x_test = np.array(x_test)
        elif model_name == 'XGBoost':
            print('XGBoost model')
            model = xgb.XGBClassifier(objective="binary:logistic")
        elif model_name == 'SVM':
            print('SVM model')
            model = svm.SVC(kernel='rbf')
        elif model_name == 'LinearSVM':
            print('LinearSVM model')
            penalty = 'l1'
            tol = 0.1
            C = 1
            dual = False
            class_weight = None
            max_iter = 100
            model = svm.LinearSVC(penalty=penalty, tol=tol, C=C, dual=dual,
                                  class_weight=class_weight, max_iter=max_iter)
        elif model_name == 'OneSVM':
            print('OneSVM model')
            nu = 0.12
            gamma = 0.001
            model = svm.OneClassSVM(gamma=gamma, kernel='rbf', nu=nu)
        elif model_name == 'SGDoneSVM':
            print('SGDoneSVM model')
            nu = 0.12
            tol = 1e-6
            gamma = 0.001
            transform = Nystroem(gamma=gamma, random_state=SEED)
            clf_sgd = SGDOneClassSVM(nu=nu, shuffle=True, fit_intercept=True, random_state=SEED, tol=tol)
            model = make_pipeline(transform, clf_sgd)
        elif model_name == 'IsolationForest':
            print('IsolationForest model')
            model = IsolationForest(n_estimators=10, warm_start=True)
        elif model_name == 'RandomForest':
            print('RandomForest model')
            model = RandomForestClassifier(max_depth=2, random_state=0)

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        if model_name == 'OneSVM' or model_name == 'IsolationForest' or model_name == 'SGDoneSVM':
            y_pred = [1 if i == -1 else 0 for i in y_pred]
        # print(y_pred)

        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        # print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))

        models_results[model_name] = {'precision': precision, 'recall': recall, 'f1': f1}

    return models_results


models = ['CatBoost', 'XGBoost', 'RandomForest', 'IsolationForest', 'SVM', 'OneSVM', 'LinearSVM', 'SGDoneSVM']

hdfs_root = '/Users/d.volf/Documents/Projects/log_ml/benchmark_datasets/loglizer-master/data/HDFS'
hdfs_log = os.path.join(hdfs_root, 'HDFS_100k.log_structured.csv')
hdfs_label = os.path.join(hdfs_root, 'anomaly_label.csv')

# hdfs_log = os.path.join('../data/HDFS/', 'HDFS.npz')


# module_url = 'https://tfhub.dev/google/universal-sentence-encoder/4'
# USE_embed = hub.KerasLayer(module_url, trainable=False, name='USE_embedding')
USE_embed = hub.load(
    '/Users/d.volf/Documents/Projects/log_ml/alphabeaver-ml/notebooks/use_utils/universal-sentence-encoder_4')

# (x_train, y_train), (x_test, y_test) = my_dataloader.load_HDFS(hdfs_log, window='session', train_ratio=0.5,
#                                                                split_type='uniform')

(x_train, y_train), (x_test, y_test) = my_dataloader.load_HDFS(hdfs_log, label_file=hdfs_label, window='session',
                                                               train_ratio=0.5, split_type='uniform',
                                                               mod='USE', clean_text_flg=False)

# join messages by block_id
x_train = join_messages(x_train)
x_test = join_messages(x_test)

print(f'x_train before: {x_train}')
print(f'x_train before: {len(x_train)}')

# transform by USE model
x_train = USE_embed(x_train)

print(f'x_train after: {x_train}')
print(f'x_train after: {x_train.shape}')

x_test = USE_embed(x_test)

x_train = np.array(x_train)
x_test = np.array(x_test)

10/0

# TF-IDF from loglizer
# feature_extractor = my_tf_idf.FeatureExtractor()
# x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')
# x_test = feature_extractor.transform(x_test)
# [('SGDoneSVM', {'precision': 0.11463452195948311, 'recall': 0.9725620620026132, 'f1': 0.20509480750444609}),
# ('IsolationForest', {'precision': 0.25347675269706904, 'recall': 0.9460743556241834, 'f1': 0.39982932583705644}),
# ('OneSVM', {'precision': 0.3965709304015111, 'recall': 0.9725620620026132, 'f1': 0.5634074176013212}),
# ('RandomForest', {'precision': 0.9991374353076481, 'recall': 0.41275685948449936, 'f1': 0.5841808859376314}),
# ('CatBoost', {'precision': 0.9771108850457783, 'recall': 0.9126974700083146, 'f1': 0.943806423877664}),
# ('LinearSVM', {'precision': 0.9464392222354766, 'recall': 0.9423922081007245, 'f1': 0.9444113795976671}),
# ('SVM', {'precision': 0.9923366986984552, 'recall': 0.9689986934315239, 'f1': 0.9805288461538462}),
# ('XGBoost', {'precision': 0.9971537001897534, 'recall': 0.998693431523934, 'f1': 0.9979229719304492})]

# TF-IDF from sklearn
vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(x_train).toarray()
vectorizer.get_feature_names_out()
x_test = vectorizer.transform(x_test).toarray()
# [('OneSVM', {'precision': 0.035771606974034253, 'recall': 0.9662667775270222, 'f1': 0.06898920427080064}),
# ('IsolationForest', {'precision': 0.1646695244741302, 'recall': 0.9809953676208576, 'f1': 0.28200225355959985}),
# ('SGDoneSVM', {'precision': 0.3920792079207921, 'recall': 0.9642475353367383, 'f1': 0.5574783683559951}),
# ('RandomForest', {'precision': 0.9991286668602962, 'recall': 0.40859959615156194, 'f1': 0.5800033721126285}),
# ('CatBoost', {'precision': 0.894896648370898, 'recall': 0.9102031120085521, 'f1': 0.9024849841008126}),
# ('LinearSVM', {'precision': 0.9602474510253179, 'recall': 0.9956051787623234, 'f1': 0.9776067179846046}),
# ('SVM', {'precision': 0.9931222577967509, 'recall': 0.9947737260957359, 'f1': 0.9939473059577499}),
# ('XGBoost', {'precision': 0.9972713251868549, 'recall': 0.9984558736191946, 'f1': 0.997863247863248})])

# TF-IDF from sklearn + scaler = StandardScaler()
# [('OneSVM', {'precision': 0.11582379484921858, 'recall': 1.0, 'f1': 0.20760230313043262}),
# ('IsolationForest', {'precision': 0.17271869358279993, 'recall': 1.0, 'f1': 0.294561167188566}),
# ('RandomForest', {'precision': 0.9991291727140784, 'recall': 0.4088371540563012, 'f1': 0.5802427511800404}),
# ('SGDoneSVM', {'precision': 0.4386038030737171, 'recall': 1.0, 'f1': 0.6097631636126603}),
# ('CatBoost', {'precision': 0.8887858054041516, 'recall': 0.9103218909609218, 'f1': 0.899424950123225}),
# ('LinearSVM', {'precision': 0.9605866177818515, 'recall': 0.9958427366670626, 'f1': 0.9778970082230127}),
# ('SVM', {'precision': 0.9943208707998107, 'recall': 0.9982183157144554, 'f1': 0.9962657815185821}),
# ('XGBoost', {'precision': 0.9972713251868549, 'recall': 0.9984558736191946, 'f1': 0.997863247863248})]

# scale
scaler = StandardScaler()
# scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

models_res = models_cheker(x_train, y_train, x_test, y_test, models=models)

models_res = OrderedDict(sorted(models_res.items(), key=lambda x: (x[1]['f1'])))
print(models_res)

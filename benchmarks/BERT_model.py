import os
import sys
import numpy as np
import pandas as pd

import time

import my_dataloader
from sklearn.metrics import precision_recall_fscore_support

import tensorflow as tf
print(tf.__version__)

import tensorflow_hub as hub

import xgboost as xgb

import matplotlib.pyplot as plt

import json

from official.nlp import bert

# Load the required submodules
import official.nlp.optimization
import official.nlp.bert.bert_models
import official.nlp.bert.configs
import official.nlp.bert.run_classifier
import official.nlp.bert.tokenization
import official.nlp.data.classifier_data_lib
import official.nlp.modeling.losses
import official.nlp.modeling.models
import official.nlp.modeling.networks

from sentence_transformers import SentenceTransformer

sys.path.append('../')
verbose = 1
SEED = 42


def join_messages(X):
    x_trans = np.asarray(['. '.join(x) for x in X])
    return x_trans

hdfs_root = '/Users/d.volf/Documents/Projects/log_ml/benchmark_datasets/loglizer-master/data/HDFS'
hdfs_log = os.path.join(hdfs_root, 'HDFS_100k.log_structured.csv')
hdfs_label = os.path.join(hdfs_root, 'anomaly_label.csv')

# hdfs_log = os.path.join('../data/HDFS/', 'HDFS.npz')

# hub_url_bert = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3"
# gs_folder_bert = "gs://cloud-tpu-checkpoints/bert/v3/uncased_L-12_H-768_A-12"
# tf.io.gfile.listdir(gs_folder_bert)

# (x_train, y_train), (x_test, y_test) = my_dataloader.load_HDFS(hdfs_log, window='session', train_ratio=0.5,
#                                                                split_type='uniform')

(x_train, y_train), (x_test, y_test) = my_dataloader.load_HDFS(hdfs_log, label_file=hdfs_label, window='session',
                                                               train_ratio=0.5, split_type='uniform',
                                                               mod='USE', clean_text_flg=False)

# join messages by block_id
x_train = join_messages(x_train)
x_test = join_messages(x_test)

# bert_config_file = os.path.join(gs_folder_bert, "bert_config.json")
# config_dict = json.loads(tf.io.gfile.GFile(bert_config_file).read())
#
# bert_config = bert.configs.BertConfig.from_dict(config_dict)
# print(config_dict)
#
# bert_classifier, bert_encoder = bert.bert_models.classifier_model(bert_config, num_labels=2)
#
# tf.keras.utils.plot_model(bert_classifier, show_shapes=True, dpi=48)

start = time.time()
# model = SentenceTransformer('stsb-roberta-large')

model = SentenceTransformer('all-MiniLM-L6-v2')

print(f'x_train {type(x_train)}')
print(f'x_train {x_train.shape}')
print(f'x_train {x_train[:3]}')

10/0

x_train_embeddings = model.encode(x_train)
x_test_embeddings = model.encode(x_test)
print(time.time() - start)

# BERT doesn't have feature names
model = xgb.XGBClassifier(objective="binary:logistic")
model.fit(x_train_embeddings, y_train)

y_pred = model.predict(x_test_embeddings)

precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))

# Precision: 0.947, recall: 0.459, F1-measure: 0.618 - 'all-MiniLM-L6-v2'
# Precision: 1.000, recall: 0.439, F1-measure: 0.611 - 'stsb-roberta-large' - 7836.83562374115 seconds for embedding
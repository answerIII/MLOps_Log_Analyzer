import os
import sys
import numpy as np

import my_dataloader
from sklearn.metrics import precision_recall_fscore_support

import tensorflow as tf
import tensorflow_hub as hub

import matplotlib.pyplot as plt

sys.path.append('../')
verbose = 1
SEED = 42


def join_messages(X):
    x_trans = np.asarray(['. '.join(x) for x in X])
    return x_trans

def build_model(embed, output_size):
    #     model = tf.keras.models.Sequential()
    #     model.add(hub.KerasLayer(USE_embed, input_shape=[], dtype=tf.string))
    # #     model.add(tf.keras.layers.Dense(256, activation='relu'))
    #     model.add(tf.keras.layers.Dense(output_size, activation='sigmoid'))
    #     model.compile(optimizer='adam', loss='categorical_crossentropy',
    #                   metrics=[tf.keras.metrics.PrecisionAtRecall(recall=0.7)])

    model = tf.keras.models.Sequential()
    model.add(hub.KerasLayer(USE_embed, input_shape=[], dtype=tf.string))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(output_size, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# hdfs_root = '/Users/d.volf/Documents/Projects/log_ml/benchmark_datasets/loglizer-master/data/HDFS'
# hdfs_log = os.path.join(hdfs_root, 'HDFS_100k.log_structured.csv')
# hdfs_label = os.path.join(hdfs_root, 'anomaly_label.csv')

hdfs_log = os.path.join('../data/HDFS/', 'HDFS.npz')


# module_url = 'https://tfhub.dev/google/universal-sentence-encoder/4'
# USE_embed = hub.KerasLayer(module_url, trainable=False, name='USE_embedding')
USE_embed = hub.load(
    '/Users/d.volf/Documents/Projects/log_ml/alphabeaver-ml/notebooks/use_utils/universal-sentence-encoder_4')

(x_train, y_train), (x_test, y_test) = my_dataloader.load_HDFS(hdfs_log, window='session', train_ratio=0.5,
                                                               split_type='uniform')

# (x_train, y_train), (x_test, y_test) = my_dataloader.load_HDFS(hdfs_log, label_file=hdfs_label, window='session',
#                                                                train_ratio=0.5, split_type='uniform',
#                                                                mod='USE', clean_text_flg=False)

# join messages by block_id
x_train = join_messages(x_train)
x_test = join_messages(x_test)

# Transfer learning
output_size = 1
model = build_model(USE_embed, output_size=output_size)
model.summary()

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=4, verbose=1,
                                      mode='min', baseline=None, restore_best_weights=True)

history = model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1,
                    validation_data=(np.asarray(x_test), y_test))

oof_preds = np.zeros([np.asarray(x_train).shape[0], output_size])
plot_metrics = ['loss', 'accuracy']

best_index = np.argmin(history.history['val_loss'])
oof_preds = model.predict(x_train)

f, ax = plt.subplots(1, len(plot_metrics))
for p_i, metric in enumerate(plot_metrics):
    ax[p_i].plot(history.history[metric], label='Train ' + metric)
    ax[p_i].plot(history.history['val_' + metric], label='Val ' + metric)
    ax[p_i].set_title("Loss Curve - {}\nBest Epoch {}".format(metric, best_index))
    ax[p_i].legend()
    ax[p_i].axvline(x=best_index, c='black')
plt.show()

model.predict(x_test, verbose=0)
y_classes = model.predict_classes(x_test, verbose=0)

precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_classes, average='binary')
print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))

# Precision: 0.985, recall: 0.427, F1-measure: 0.596
# Precision: 0.993, recall: 0.979, F1-measure: 0.986 - full dataset

print()

# reducer = UMAP(n_neighbors=100, n_components=3, metric='euclidean', n_epochs=1000, learning_rate=1.0,
#                init='spectral', min_dist=0.1, spread=1.0, low_memory=False, set_op_mix_ratio=1.0,
#                local_connectivity=1, repulsion_strength=1.0, negative_sample_rate=5, transform_queue_size=4.0,
#                a=None, b=None, random_state=42, metric_kwds=None, angular_rp_forest=False,
#                target_n_neighbors=-1, transform_seed=42, verbose=False, unique=False)
#
# # Fit and transform the data
# X_trans = reducer.fit_transform(x_train_embeddings)
#
#
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import axes3d
#
# X_trans_normal = []
# X_trans_anomaly = []
# for ind, y in enumerate(y_train):
#     if y == 1:
#         X_trans_anomaly.append(X_trans[ind])
#     else:
#         X_trans_normal.append(X_trans[ind])
#
# X_trans_anomaly = np.array(X_trans_anomaly)
# X_trans_normal = np.array(X_trans_normal)
#
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X_trans_anomaly[:, 0], X_trans_anomaly[:, 1], X_trans_anomaly[:, 2], c='b')
# ax.scatter(X_trans_normal[:, 0], X_trans_normal[:, 1], X_trans_normal[:, 2], c='r')
# plt.show()
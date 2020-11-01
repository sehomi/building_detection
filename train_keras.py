#!/usr/bin/env python

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os
import tempfile
import seaborn as sns

mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


data = np.loadtxt("dataset/association/features/data1.csv", delimiter=",")

train_data, test_data = train_test_split(data, test_size=0.2)
train_data, val_data = train_test_split(train_data, test_size=0.2)

trainy = np.array(train_data[:,16])
bool_trainy = trainy != 0
valy = np.array(val_data[:,16])
testy = np.array(test_data[:,16])

trainX = np.array(train_data[:,0:16])
valX = np.array(val_data[:,0:16])
testX = np.array(test_data[:,0:16])

scaler = StandardScaler()
trainX = scaler.fit_transform(trainX)

pickle.dump(scaler, open('scaler.pkl','wb'))

valX = scaler.transform(valX)
testX = scaler.transform(testX)

trainX = np.clip(trainX, -5, 5)
valX = np.clip(valX, -5, 5)
testX = np.clip(testX, -5, 5)


print('Training labels shape:', trainy.shape)
print('Validation labels shape:', valy.shape)
print('Test labels shape:', testy.shape)

print('Training features shape:', trainX.shape)
print('Validation features shape:', valX.shape)
print('Test features shape:', testX.shape)

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
]

def plot_loss(history, label, n):
  # Use a log scale to show the wide range of values.
  plt.semilogy(history.epoch,  history.history['loss'],
               color=colors[n], label='Train '+label)
  plt.semilogy(history.epoch,  history.history['val_loss'],
          color=colors[n], label='Val '+label,
          linestyle="--")
  plt.xlabel('Epoch')
  plt.ylabel('Loss')

  plt.legend()

def plot_roc(name, labels, predictions, **kwargs):
  fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

  plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
  plt.xlabel('False positives [%]')
  plt.ylabel('True positives [%]')
  plt.xlim([-0.5,20])
  plt.ylim([80,100.5])
  plt.grid(True)
  ax = plt.gca()
  ax.set_aspect('equal')


def plot_metrics(history):
  metrics =  ['loss', 'auc', 'precision', 'recall']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch,  history.history[metric], color=colors[0], label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
             color=colors[0], linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.8,1])
    else:
      plt.ylim([0,1])

    plt.legend()

def plot_cm(labels, predictions, p=0.5):
  cm = confusion_matrix(labels, predictions > p)
  plt.figure(figsize=(5,5))
  sns.heatmap(cm, annot=True, fmt="d")
  plt.title('Confusion matrix @{:.2f}'.format(p))
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')

  print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
  print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
  print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
  print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
  print('Total Fraudulent Transactions: ', np.sum(cm[1]))


def make_model(metrics = METRICS, output_bias=None):
  if output_bias is not None:
     output_bias = tf.keras.initializers.Constant(output_bias)
     model = keras.Sequential([
          keras.layers.Dense(
              32, activation='relu',
              input_shape=(trainX.shape[-1],)),
          keras.layers.Dropout(0.5),
          keras.layers.Dense(1, activation='sigmoid',
                             bias_initializer=output_bias),
      ])
  else:
     model = keras.Sequential([
          keras.layers.Dense(
              64, activation='relu',
              input_shape=(trainX.shape[-1],)),
          keras.layers.Dropout(0.5),
          keras.layers.Dense(1, activation='sigmoid'),
      ])

  model.compile(
      optimizer=keras.optimizers.Adam(lr=1e-3),
      loss="binary_crossentropy",
      metrics=metrics)

  return model

EPOCHS = 100
BATCH_SIZE = 1024

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc',
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)

pos = np.count_nonzero(data[:,16] == 1)
neg = np.count_nonzero(data[:,16] == 0)
total = data.shape[0]

weight_for_0 = (1 / neg)*(total)/2.0
weight_for_1 = (1 / pos)*(total)/2.0


class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))


b0 = np.log([pos/neg])
print(pos, neg, b0)

model = make_model(output_bias=b0)
model.summary()

initial_weights = os.path.join(tempfile.mkdtemp(),'initial_weights')
model.save_weights(initial_weights)


model.load_weights(initial_weights)
baseline_history = model.fit(
    trainX,
    trainy,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks = [early_stopping],
    validation_data=(valX, valy),
    # class_weight=class_weight)
    )


# plot_loss(baseline_history, "", 0)
plot_metrics(baseline_history)

test_predictions_baseline = model.predict(testX, batch_size=BATCH_SIZE)
train_predictions_baseline = model.predict(trainX, batch_size=BATCH_SIZE)

plot_cm(testy, test_predictions_baseline)
# plot_roc("Train Baseline", trainy, train_predictions_baseline, color=colors[0])
# plot_roc("Test Baseline", testy, test_predictions_baseline, color=colors[0], linestyle='--')
plt.legend(loc='lower right')


baseline_results = model.evaluate(testX, testy,
                                  batch_size=BATCH_SIZE, verbose=0)
for name, value in zip(model.metrics_names, baseline_results):
  print(name, ': ', value)
print()

model.save('model')

plt.show()

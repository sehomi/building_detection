#!/usr/bin/env python

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import classification_report, plot_confusion_matrix, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
import pickle
import os
import seaborn as sns

mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


data = np.loadtxt("dataset/association/features/data4.csv", delimiter=",")

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

pickle.dump(scaler, open('models/scaler5.pkl','wb'))

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

gnb = GaussianNB()
clf = gnb.fit(trainX, trainy)


preds = gnb.predict(testX) 
preds_train = gnb.predict(trainX) 

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

  plt.savefig('cm_naive.pdf', format='PDF')

plot_cm(testy, preds)


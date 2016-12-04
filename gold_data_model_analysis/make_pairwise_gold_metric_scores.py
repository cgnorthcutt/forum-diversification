
# coding: utf-8

# In[ ]:

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function


# In[5]:

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import sys


# In[1]:

def compute_metrics(pairwise_cosine_similarity_train, pairwise_cosine_similarity_test, gold_matrix_train, gold_data_train, gold_matrix_test, gold_data_test):
  
  # Scale / Normalize pairwise_cosine_similarity
  pairwise_cosine_similarity_test = MinMaxScaler().fit_transform(pairwise_cosine_similarity_test)
  
  # Convert gold matrices to np.array (instead of pd.DataFrame) if necessary
  gold_matrix_train = gold_matrix_train.values if isinstance(gold_matrix_train, pd.DataFrame) else gold_matrix_train
  gold_matrix_test = gold_matrix_test.values if isinstance(gold_matrix_test, pd.DataFrame) else gold_matrix_test
  
  # Compute list of indices of sorted matrix values in list form
  flat_cossim_train = pairwise_cosine_similarity_train.flatten()
  flat_cossim_test = pairwise_cosine_similarity_test.flatten()
  flat_gold_train = gold_matrix_train.flatten()
  flat_gold_test = gold_matrix_test.flatten()
  
  npairs_train = float(len(flat_cossim_train))
  npairs_test = float(len(flat_cossim_test))
  
  metrics = {}
  
  #
  # Metric 1: avg_diff (Average difference of cossine similarity for gold score classes)
  #
  
  # Compute avg consine similarity for same cluster comments and different cluster topics and subtract.
  same_cluster_avg_score = np.multiply(pairwise_cosine_similarity_test, gold_matrix_test).sum() / gold_matrix_test.sum()
  diff_cluster_avg_score = np.multiply(pairwise_cosine_similarity_test, 1-gold_matrix_test).sum() / (1-gold_matrix_test).sum()
  metrics["avg_diff"] = same_cluster_avg_score - diff_cluster_avg_score
  print("Avg Difference score:", same_cluster_avg_score, "-", diff_cluster_avg_score, "=", metrics["avg_diff"])
  
  #
  # Metric 1: median_diff (Median difference of cossine similarity for gold score classes)
  #
  
  gold0_values = flat_cossim_test[flat_gold_test == 0]
  gold1_values = flat_cossim_test[flat_gold_test == 1]
  np.median(gold1_values) - np.median(gold0_values)
  metrics["median_diff"] = np.median(gold1_values) - np.median(gold0_values)
  print("Median Difference score:", np.median(gold1_values), "-", np.median(gold0_values), "=", metrics["median_diff"])
  
  #
  # Metric 2: median_quantile_diff (Median normalized rank difference of cossine similarity for gold score classes)
  #
  
  ranks = rankdata(flat_cossim_test)
  gold0_values_quantile = ranks[flat_gold_test == 0] / npairs_test
  gold1_values_quantile = ranks[flat_gold_test == 1] / npairs_test
  np.median(gold1_values_quantile) - np.median(gold0_values_quantile)
  metrics["median_quantile_diff"] = np.median(gold1_values_quantile) - np.median(gold0_values_quantile)
  print("Median Quantile (Rank) Difference score:", np.median(gold1_values_quantile), "-", np.median(gold0_values_quantile), "=", metrics["median_quantile_diff"])
  sys.stdout.flush()
  
  #
  # Metric 3: logreg_acc_pairwise_binary Binary Logistic Regression Accuracy
  # 
  
  clf = LogisticRegression()
  clf.fit(flat_cossim_train.reshape((int(npairs_train), 1)), flat_gold_train.reshape((int(npairs_train), 1)))
  y_pred = clf.predict(flat_cossim_test.reshape((int(npairs_test), 1)))
  acc = accuracy_score(y_true = flat_gold_test, y_pred = y_pred)
  metrics["logreg_acc_pairwise_binary"] = acc
  print("Pairwise Binary Logistic Regression Accuracy score:", acc)
  sys.stdout.flush()
  
#   #
#   # Metric 4: logreg_acc_topic Binary Logistic Regression Accuracy
#   # 
  
#   clf = LogisticRegression()
#   clf.fit(flat_cossim_train, gold_data)
#   y_pred = clf.pred(flat_cossim_test)
#   acc = accuracy_score(y_true = flat_gold, y_pred = y_pred)
#   metrics["logreg_acc_binary"] = acc
#   print("Binary Logistic Regression Accuracy score:", acc)
#   sys.stdout.flush()
  
  return metrics


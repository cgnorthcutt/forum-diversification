
# coding: utf-8

# In[9]:

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function


# In[16]:

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from make_pairwise_gold_metric_scores import compute_metrics

pd.set_option('max_colwidth',5000)


# In[17]:

# Load word2vec model for this specific course
w2v_matrix = pickle.load( open( "../data/w2v_matrix_for_HarvardX__HDS_3221_2X__1T2016.p", "rb" ) )
vocab = np.array(pickle.load( open( "../data/vocab_for_HarvardX__HDS_3221_2X__1T2016.p", "rb" ) ) )


# In[18]:

from sklearn.decomposition import PCA


# In[19]:

def tfidf_lsa(gold_data, gold_matrix, eigenvector=20):
  # Get tfidf counts for each comment as a matrix C with shape (# comments, size of vocab)
  tfidf = TfidfVectorizer(stop_words='english')
  C = tfidf.fit_transform(gold_data.body.values)
  print("tfidf C shape", C.shape)
  lsa = TruncatedSVD(eigenvector, algorithm = 'arpack')
  A = lsa.fit_transform(C)
  print("LSA shape", A.shape)
  A=normalize(A,norm="l2")
  # We compute pairwise cosine similarity with dot product since A is normalized.
  pairwise_cosine_similarity = np.dot(A, A.transpose())
  return MinMaxScaler().fit_transform(pairwise_cosine_similarity)


# In[20]:

# Load gold train data
gold_matrix_train = pd.read_csv('gold_matrix_train_HarvardX__HDS_3221_2X__1T2016.csv.gz', compression='gzip')
df_gold_train = pd.read_csv('gold_data_train_HarvardX__HDS_3221_2X__1T2016.csv.gz', compression='gzip')
pairwise_cosine_similarity_train = tfidf_lsa(df_gold_train, gold_matrix_train)


# In[21]:

# Load gold data
gold_matrix_test = pd.read_csv('gold_matrix_test_HarvardX__HDS_3221_2X__1T2016.csv.gz', compression='gzip')
df_gold_test = pd.read_csv('gold_data_test_HarvardX__HDS_3221_2X__1T2016.csv.gz', compression='gzip')
pairwise_cosine_similarity_test = tfidf_lsa(df_gold_test, gold_matrix_test)


# In[22]:

metrics = compute_metrics(pairwise_cosine_similarity_train, pairwise_cosine_similarity_test, gold_matrix_train, df_gold_train, gold_matrix_test, df_gold_test)
pretty_metrics = pd.DataFrame(pd.Series(metrics), columns = ["Score"])
pretty_metrics


# In[23]:

# Switching train and test

metrics = compute_metrics(pairwise_cosine_similarity_test, pairwise_cosine_similarity_train, gold_matrix_test, df_gold_test, gold_matrix_train, df_gold_train)
pretty_metrics = pd.DataFrame(pd.Series(metrics), columns = ["Score"])
pretty_metrics


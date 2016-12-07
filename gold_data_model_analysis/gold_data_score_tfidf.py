
# coding: utf-8

# In[1]:

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function


# In[2]:

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import rankdata

pd.set_option('max_colwidth',5000)


# In[3]:

# Load word2vec model for this specific course
w2v_matrix = pickle.load( open( "../data/w2v_matrix_for_HarvardX__HDS_3221_2X__1T2016.p", "rb" ) )
vocab = np.array(pickle.load( open( "../data/vocab_for_HarvardX__HDS_3221_2X__1T2016.p", "rb" ) ) )


# In[4]:

# Verify word2vec model works well (the following example should have cosine similarity near 1)

# def cosinesimilarity(u, v):
#   # u - embedding (vector)
#   # v - embedding (vector)
#   return np.dot(u,v) / (np.sqrt(np.dot(u, u)) * np.sqrt(np.dot(v,v)))
# np.dot(w2v_matrix[np.where(vocab == "queen")[0][0]], w2v_matrix[np.where(vocab == "king")[0][0]] - w2v_matrix[np.where(vocab == "man")[0][0]] + w2v_matrix[np.where(vocab == "woman")[0][0]] )


# In[5]:

test_comments = [
  "Jesus is great. I love Jesus and the holy book. I think God is amazing. I agree with religion",
  "Jesus is great. I love Jesus and the holy book. I think God is amazing.",
  "Jesus is the best. I love Jesus and the holy book. I think God is amazing. I agree with religion",
  "Jesus is awesome. I love Jesus and the holy book. I think God is amazing. I agree with religion",
  "Christ is great. I love Christ and the holy book. I think God is amazing. I agree with religion",
  "Christ is great. I love Christ and the holy book. I agree with religion",
  "Donald Trump is a president with tons of electrical equipment and lightbulbs and random things.",
]


# In[56]:

TfidfVectorizer(stop_words='english', )


# In[57]:

def tfidf(gold_data, gold_matrix):
  # Get tfidf counts for each comment as a matrix C with shape (# comments, size of vocab)
  vec = TfidfVectorizer(stop_words='english')
  C = vec.fit_transform(gold_data.body.values)
  C = normalize(C, norm='l2')
  # We compute pairwise cosine similarity with dot product since A is normalized.
  pairwise_cosine_similarity = np.dot(C, C.transpose())
  return MinMaxScaler().fit_transform(pairwise_cosine_similarity.toarray())


# In[58]:

# Load gold train data
gold_matrix_train = pd.read_csv('gold_matrix_train_HarvardX__HDS_3221_2X__1T2016.csv.gz', compression='gzip')
df_gold_train = pd.read_csv('gold_data_train_HarvardX__HDS_3221_2X__1T2016.csv.gz', compression='gzip')
pairwise_cosine_similarity_train = tfidf(df_gold_train, gold_matrix_train)


# In[59]:

# Load gold data
gold_matrix_test = pd.read_csv('gold_matrix_test_HarvardX__HDS_3221_2X__1T2016.csv.gz', compression='gzip')
df_gold_test = pd.read_csv('gold_data_test_HarvardX__HDS_3221_2X__1T2016.csv.gz', compression='gzip')
pairwise_cosine_similarity_test = tfidf(df_gold_test, gold_matrix_test)


# In[43]:

import make_pairwise_gold_metric_scores
reload(make_pairwise_gold_metric_scores)
from make_pairwise_gold_metric_scores import compute_metrics


# In[60]:

metrics = compute_metrics(pairwise_cosine_similarity_train, pairwise_cosine_similarity_test, gold_matrix_train, df_gold_train, gold_matrix_test, df_gold_test)
pretty_metrics = pd.DataFrame(pd.Series(metrics), columns = ["Score"])
pretty_metrics


# In[45]:

# Switching train and test

metrics = compute_metrics(pairwise_cosine_similarity_test, pairwise_cosine_similarity_train, gold_matrix_test, df_gold_test, gold_matrix_train, df_gold_train)
pretty_metrics = pd.DataFrame(pd.Series(metrics), columns = ["Score"])
pretty_metrics


# In[ ]:




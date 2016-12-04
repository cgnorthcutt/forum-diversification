
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


# In[10]:

def embedding_tfidf(embedding_matrix, embedding_vocab, gold_data, gold_matrix, test_name):
  # Get tfidf counts for each comment as a matrix C with shape (# comments, size of w2v (embedding) vocab)
  vec = TfidfVectorizer(vocabulary=embedding_vocab)
  C = vec.fit_transform(gold_data.body.values)

  # Compute tfidf bag of words for each comment as matrix A with shape (#comments, embedding dimension)
  A = C.dot(embedding_matrix)
  A = normalize(A, norm='l2')

  # Verify that each row of A is normalized (unit vector l2-norm)
  assert(abs(sum(np.sum(np.abs(A)**2,axis=-1)**(1./2)) / len(A) - 1.0) < 0.0001) # Should be close to 1.0

  # We compute pairwise cosine similarity with dot product since A is normalized.
  pairwise_cosine_similarity = np.dot(A, A.transpose())
  pairwise_cosine_similarity = MinMaxScaler().fit_transform(pairwise_cosine_similarity)

  # Compute avg consine similarity for same cluster comments and different cluster topics and subtract.
  same_cluster_avg_score = np.multiply(pairwise_cosine_similarity, gold_matrix).values.sum() / gold_matrix.values.sum()
  diff_cluster_avg_score = np.multiply(pairwise_cosine_similarity, 1-gold_matrix).values.sum() / (1-gold_matrix).values.sum()
  print(test_name, "score:", same_cluster_avg_score, "-", diff_cluster_avg_score, "=", same_cluster_avg_score - diff_cluster_avg_score)
  
  return pairwise_cosine_similarity


# In[11]:

# Load gold train data
gold_matrix_train = pd.read_csv('gold_matrix_train_HarvardX__HDS_3221_2X__1T2016.csv.gz', compression='gzip')
df_gold_train = pd.read_csv('gold_data_train_HarvardX__HDS_3221_2X__1T2016.csv.gz', compression='gzip')
pairwise_cosine_similarity_train = embedding_tfidf(w2v_matrix, vocab, df_gold_train, gold_matrix_train, "Train")


# In[12]:

# Load gold data
gold_matrix_test = pd.read_csv('gold_matrix_test_HarvardX__HDS_3221_2X__1T2016.csv.gz', compression='gzip')
df_gold_test = pd.read_csv('gold_data_test_HarvardX__HDS_3221_2X__1T2016.csv.gz', compression='gzip')
pairwise_cosine_similarity_test =embedding_tfidf(w2v_matrix, vocab, df_gold_test, gold_matrix_test, "Test")


# In[41]:

from make_pairwise_gold_metric_scores import compute_metrics

metrics = compute_metrics(pairwise_cosine_similarity_train, pairwise_cosine_similarity_test, gold_matrix_train, df_gold_train, gold_matrix_test, df_gold_test)
pretty_metrics = pd.DataFrame(pd.Series(metrics), columns = ["Score"])
pretty_metrics


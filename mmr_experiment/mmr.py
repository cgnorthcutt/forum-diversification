
# coding: utf-8

# In[4]:

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function


# In[ ]:

import numpy as np
import pandas as pd
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


# In[2]:

def tfidf_pca(comments_np_array, eigenvector=20):
  # Get tfidf counts for each comment as a matrix C with shape (# comments, size of vocab)
  tfidf = TfidfVectorizer(stop_words='english')
  C = tfidf.fit_transform(comments_np_array)
#   print("tfidf C shape", C.shape)
  pca = PCA(n_components=eigenvector)
  A = pca.fit_transform(C.toarray())
#   print("PCA shape", A.shape)
  A=normalize(A,norm="l2")
  # We compute pairwise cosine similarity with dot product since A is normalized.
  pairwise_cosine_similarity = np.dot(A, A.transpose())
  return MinMaxScaler().fit_transform(pairwise_cosine_similarity)


# In[3]:

def mmr(pairwise_matrix, w, K, lam):
  '''
  pairwise_matrix - precomputed cosine similarity for all pairs of comments
  w - weights for each document which rank its goodness
  K - number of diverse documents to select
  lam - lambda tradeoff between score (weight) and diversity
  '''

  VERY_NEGATIVE_NUMBER = -1e10
  VERY_LARGE_NUMBER = 1e10

  # Init
  N = len(pairwise_matrix)
  mu_i = np.zeros(K, dtype=np.int) #index of selected comments 
  weights = np.copy(w) 
  P = np.copy(pairwise_matrix)

  #Make highest-weighted datum first selected document
  mu_i[0] = np.argmax(weights)

  #Update mask (remove doc from set) and set weight to neg inf
  weights[mu_i[0]] = VERY_NEGATIVE_NUMBER
  prev_c = np.empty(N) #Stores the max cossim across previous docs for each of the N docs
  prev_c.fill(VERY_NEGATIVE_NUMBER) #Initialization to a value that will never be max
  P[mu_i[0]].fill(VERY_LARGE_NUMBER) #effectively remove pairwise row of selected docs

  #MMR algorithm
  for k in range(1, K):

      #Reduce the computation to O(KN) instead of O(K2N) by dynamically building max.
      P_k = P[:,mu_i[k-1]] #All cosine similarities for (k-1)th doc
      prev_max_concat_new_cossim = np.column_stack((prev_c, P_k)) #shape is (N, 2)
      c = np.max(prev_max_concat_new_cossim, axis = 1) #max cosine sim b
      prev_c = c #update dynamic max cossim for each doc across all selected docs
      
      #Compute MMR scores for each document
      #Scores for previously selected documents will remain VERY_NEGATIVE_NUMBER
      scores = lam * weights - (1-lam) * c

      #Select document with maximum score
      mu_i[k] = np.argmax(scores)

      #Update mask (remove doc from set) and set weight to neg inf
      weights[mu_i[k]] = VERY_NEGATIVE_NUMBER
      P[mu_i[k]].fill(VERY_LARGE_NUMBER) #effectively remove pairwise row of selected doc

  return mu_i



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

pd.set_option('max_colwidth',5000)


# In[3]:

# Load gold data
gold_matrix = pd.read_csv('gold_matrix_for_HarvardX__HDS_3221_2X__1T2016.csv.gz', compression='gzip')
df_gold = pd.read_csv('gold_data_HarvardX__HDS_3221_2X__1T2016.csv.gz', compression='gzip')

# Load word2vec model for this specific course
w2v_matrix = pickle.load( open( "w2v_matrix_for_HarvardX__HDS_3221_2X__1T2016.p", "rb" ) )
vocab = np.array(pickle.load( open( "vocab_for_HarvardX__HDS_3221_2X__1T2016.p", "rb" ) ) )


# In[117]:

# Verify word2vec model works well (the following example should have cosine similarity near 1)

# def cosinesimilarity(u, v):
#   # u - embedding (vector)
#   # v - embedding (vector)
#   return np.dot(u,v) / (np.sqrt(np.dot(u, u)) * np.sqrt(np.dot(v,v)))
# np.dot(w2v_matrix[np.where(vocab == "queen")[0][0]], w2v_matrix[np.where(vocab == "king")[0][0]] - w2v_matrix[np.where(vocab == "man")[0][0]] + w2v_matrix[np.where(vocab == "woman")[0][0]] )


# In[4]:

test_comments = [
  "Jesus is great. I love Jesus and the holy book. I think God is amazing. I agree with religion",
  "Jesus is great. I love Jesus and the holy book. I think God is amazing.",
  "Jesus is the best. I love Jesus and the holy book. I think God is amazing. I agree with religion",
  "Jesus is awesome. I love Jesus and the holy book. I think God is amazing. I agree with religion",
  "Christ is great. I love Christ and the holy book. I think God is amazing. I agree with religion",
  "Christ is great. I love Christ and the holy book. I agree with religion",
  "Donald Trump is a president with tons of electrical equipment and lightbulbs and random things.",
]


# In[7]:

# Get tfidf counts for each comment as a matrix C with shape (# comments, size of w2v vocab)
vec = TfidfVectorizer(vocabulary=vocab)
C = vec.fit_transform(df_gold.body.values)

# Compute tfidf bag of words for each comment as matrix A with shape (#comments, embedding dimension)
A = C.dot(w2v_matrix)
A = normalize(A, norm='l2')

# Verify that each row of A is normalized (unit vector l2-norm)
assert(abs(sum(np.sum(np.abs(A)**2,axis=-1)**(1./2)) / len(A) - 1.0) < 0.0001) # Should be close to 1.0

# We compute pairwise cosine similarity with dot product since A is normalized.
pairwise_cosine_similarity = np.dot(A, A.transpose())

# Compute avg consine similarity for same cluster comments and different cluster topics and subtract.
same_cluster_avg_score = np.multiply(pairwise_cosine_similarity, gold_matrix).values.sum() / gold_matrix.values.sum()
diff_cluster_avg_score = np.multiply(pairwise_cosine_similarity, 1-gold_matrix).values.sum() / (1-gold_matrix).values.sum()
print(same_cluster_avg_score, "-", diff_cluster_avg_score, "=", same_cluster_avg_score - diff_cluster_avg_score)


# In[ ]:




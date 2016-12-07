
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
from nltk.util import ngrams
import string
import re

from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors import NearestNeighbors


# In[3]:

pd.set_option('max_colwidth',5000)


# In[7]:

def remove_punctuation(text):
    return re.sub(ur"\p{P}+", "", text)


# In[8]:

def genWord2VecCommentEmbedding(w2v_model, comments_array, bigrams=True, verbose=True):
    '''Returns a matrix, each row is an embedding for a comment, each column is a dimension. Order is
 preserved.'''
 
    #First pass of TFIDF vectorizer to generate vocabulary of comments.
    #vec = HashingVectorizer(ngram_range=(1, 2), stop_words='english')
    vec = TfidfVectorizer(stop_words='english')
    C = vec.fit_transform(comments_array).toarray()
    
    if verbose:
        print("First TFIDF pass complete. Reducing comment vocabulary based on w2v table")
        
    #vocab = [i.replace(" ", "_") for i in vec.get_feature_names()]
    if bigrams:
    
        #This method computes all unique bigrams from our comments and adds them to the vocabulary
        punc = string.punctuation
        bigrams = []
        for c in comments_array:
            bigrams += [i[0] + "_" + i[1] for i in ngrams(remove_punctuation(unicode(c, 'utf-8')).split(), 2)]
        unique_bigrams = list(set(bigrams))
        vocab = vec.get_feature_names() + unique_bigrams
        
        #This method for computing bigrams finds all the bigrams which already exist in our w2v model.
        #Only add word2vec bigrams if they are in our vocab list
        #w2v_bigrams = [i for i in w2v_model.index \\
        #               if "_" in i.decode("UTF-8") and \\
        #               pd.Series(i.decode("UTF-8").split("_")).isin(vec.get_feature_names()).all()]
        #vocab = vec.get_feature_names() + w2v_bigrams
    else:
        vocab = vec.get_feature_names()
        
    W = w2v_model[w2v_model.index.isin(vocab)].sort()
    
    #Second pass of TFIDF vectorizer using vocabulary in common with w2v model and comments
    vec = TfidfVectorizer(vocabulary=W.index)
    C = vec.fit_transform(comments_array).toarray()
    W = W.values
    
    A = np.dot(C, W)
    if verbose:
        print("Shape of embedding comments matrix:",np.shape(A))
        
    return A


# In[29]:

# w2v_matrix = pickle.load( open( "/Users/cgn/Dropbox (MIT)/cgn/bq/w2v_matrix.p", "rb" ) )
# vocab = np.array(pickle.load( open( "/Users/cgn/Dropbox (MIT)/cgn/bq/word.p", "rb" ) ) )
# w2v_matrix = pickle.load( open( "w2v_matrix_for_HarvardX__HDS_3221_2X__1T2016.p", "rb" ) )
# vocab = np.array(pickle.load( open( "vocab_for_HarvardX__HDS_3221_2X__1T2016.p", "rb" ) ) )


# In[14]:

df = pd.read_csv('../data//HarvardX__HDS_3221_2X__1T2016_scored_forum_responses_and_features.csv.gz', compression='gzip')
topic_id = '5705e20881e07b74f500019d'
X = df[df.thread_id_from_url==topic_id].body.values


# In[10]:

# Find the largest n_gram.
a = [i for i in vocab if "__" not in i]
index_of_max = np.argmax([len(i.split("_")) for i in a])
a[index_of_max]


# In[17]:

comment_embeddings = genWord2VecCommentEmbedding(pd.DataFrame(w2v_matrix, index=vocab), X)


# In[18]:

# Number of clusters
k = 5

# Number of neighbors in NN model
n_neighbors = 3

# Init Nearest Neighbors model
nn = NearestNeighbors()
nn.fit(comment_embeddings)

# Init DPGMM model
dpgmm = BayesianGaussianMixture(weight_concentration_prior_type="dirichlet_process", n_components=k)
dpgmm.fit(comment_embeddings)

# Get clusters from comments
#clusters = dpgmm.predict(comment_embeddings)

# Find the nearest comment to each cluster mean 
mean_comments = df.ix[[nn.kneighbors(X=i, n_neighbors=1, return_distance=False)[0][0] for i in dpgmm.means_]]
mean_comments['cluster'] = range(k)
mean_comments


# In[157]:

kneighbors = [nn.kneighbors(X=i, n_neighbors=n_neighbors, return_distance=False)[0] for i in dpgmm.means_]
kneighbors = df.ix[[x for i in kneighbors for x in i]]
kneighbors["cluster"] = [i for i in range(k) for _ in range(n_neighbors)]
opinion_summaries = ["Bibles are different, should view all", "Diff cultures have diff books. Its okay.", "Bibles are diff because of human authorship", "Bibles are different and its okay to use different bibles.","equivocating both sides"]
kneighbors["opinion"] = [i for i in opinion_summaries for _ in range(n_neighbors)]
kneighbors


# In[70]:

df[clusters == 0] # Thinks they are different. I believe everyone should have access to all the books.


# In[71]:

df[clusters == 1] # Believes that they are the same


# In[72]:

df[clusters == 2] # They are different because of people but the message before people messed with it was the same.


# In[73]:

df[clusters == 3] # No, not important to use the same scripture. Differences are okay!


# In[74]:

df[clusters == 4] # Equivocating both sides


# In[ ]:




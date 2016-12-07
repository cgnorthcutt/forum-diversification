
# coding: utf-8

# In[2]:

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function


# In[10]:

import numpy as np
import gensim
import pickle
import pandas as pd
import string


# In[5]:

# Load Google's pre-trained Word2Vec model.
model = gensim.models.Word2Vec.load_word2vec_format('/Users/cgn/Downloads//GoogleNews-vectors-negative300.bin', binary=True)  

vocab = np.array(model.vocab.keys())
word_vector=[]
for w in vocab:
    #vector=model[w] / np.sum(model[w])
    word_vector.append(model[w])

del model
w2v_matrix=np.asarray(word_vector)


# In[8]:

# Get list of comments
df = pd.read_csv('../data/HarvardX__HDS_3221_2X__1T2016_scored_forum_responses_and_features.csv.gz', compression='gzip')
comments = list(df.body.apply(lambda x: unicode(x, 'utf-8')))


# In[12]:

# Creation of reduced w2v_matrix model with 1-gram through 6-gram

all_txt = " ".join(comments)
translate_table = dict((ord(char), None) for char in string.punctuation)   
all_words = all_txt.translate(translate_table).split()

ngram_start = 1
ngram_end = 6
all_words_ngram = ["_".join(all_words[i:i + ngram]) for ngram in range(ngram_start, ngram_end +1) for i in range(len(all_words) - (ngram - 1))]
unique_words_list = list(set(all_words_ngram))

# Build hash map from word to index
vocab_map = dict(zip(vocab, range(len(vocab))))

# Get indices of items in vocab that are contained in unique_words_list
idx_with_Nones = np.array([vocab_map.get(word) for word in unique_words_list])
idx = [i for i in idx_with_Nones if i is not None]

# Reduce w2v_matrix size by limiting to all unique words in this course.
vocab = vocab[idx]
w2v_matrix = w2v_matrix[idx]

print("Completed reduction. Dumping to pickle.")


# In[13]:

pickle.dump( w2v_matrix, open( "../data/w2v_matrix_for_HarvardX__HDS_3221_2X__1T2016.p", "wb" ) )
pickle.dump( vocab, open( "../data/vocab_for_HarvardX__HDS_3221_2X__1T2016.p", "wb" ) )


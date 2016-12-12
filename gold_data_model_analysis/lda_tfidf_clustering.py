
# coding: utf-8

# In[1]:

import sys, gensim, re, csv, operator
from __future__ import absolute_import
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from operator import itemgetter
from collections import defaultdict
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from make_pairwise_gold_metric_scores import compute_metrics

import pyLDAvis.gensim
import numpy as np
import pandas as pd
import pickle


# In[2]:

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

# Load gold training and testing data
gold_matrix_train = pd.read_csv('data/gold_matrix_train_HarvardX__HDS_3221_2X__1T2016.csv.gz', compression='gzip')
df_gold_train = pd.read_csv('data/gold_data_train_HarvardX__HDS_3221_2X__1T2016.csv.gz', compression='gzip')
gold_matrix_test = pd.read_csv('data/gold_matrix_test_HarvardX__HDS_3221_2X__1T2016.csv.gz', compression='gzip')
df_gold_test = pd.read_csv('data/gold_data_test_HarvardX__HDS_3221_2X__1T2016.csv.gz', compression='gzip')
train_resps = df_gold_train.body.values
test_resps = df_gold_test.body.values


# In[3]:

def tfidfCorpus(corpus):
  tfidf = models.TfidfModel(corpus)
  corpus_tfidf = tfidf[corpus]
  return corpus_tfidf


# In[4]:

def format_doc(document):
  # clean and tokenize document string
  raw = document.lower().decode('utf-8').strip()
  tokens = tokenizer.tokenize(raw)

  # remove stop words from tokens
  stopped_tokens = [i for i in tokens if not i in en_stop]

  # stem tokens
  stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

  # return tokens for this company
  return stopped_tokens


# In[5]:

def readFile(response_array):
  fileData = []
  rawDocs = {}
  doc_id = 1
  
  for resp in response_array:
    rawDocs[doc_id] = resp.strip()
    fileData.append(format_doc(resp))
    doc_id += 1

  return (fileData, rawDocs)


# In[6]:

def getCorpusData(response_array):
  all_tokens, rawDocs = readFile(response_array)
  
  # turn our tokenized documents into a id <-> term dictionary
  dictionary = corpora.Dictionary(all_tokens)
  
  # convert tokenized documents into a document-term matrix
  corpus = [dictionary.doc2bow(text) for text in all_tokens]
  
  return (all_tokens, rawDocs, dictionary, corpus)


# In[7]:

def topicDistsOfCps(ldaModel, dictionary, rawCps):
  cpTopicDist = {}
    
  for cp in rawCps:
    vec_bow = dictionary.doc2bow(rawCps[cp].lower().split())
    cpTopicDist[cp] = ldaModel[vec_bow]
    
  return cpTopicDist


# In[14]:

def testGold(n_topics, n_iters, doc_array, gold_matrix):
  all_tokens, rawDocs, dictionary, corpus = getCorpusData(doc_array)
  tfidf_corpus = tfidfCorpus(corpus)

  # generate LDA model with n_topics topics and n_iters iterations
  ldaModel = gensim.models.ldamodel.LdaModel(tfidf_corpus, num_topics=n_topics, id2word = dictionary, passes=n_iters)
  vis_data = pyLDAvis.gensim.prepare(ldaModel, tfidf_corpus, dictionary)

  # 
  emd = topicDistsOfCps(ldaModel, dictionary, rawDocs)
  
  # cosine similarity matrix
  M_cos = np.zeros((len(emd),len(emd)))

  for key1,value1 in emd.items():
      for key2,value2 in emd.items():
          temp = gensim.matutils.cossim(value1, value2)
          M_cos[key1-1][key2-1] = temp
            
  # largest similarity == 1
  M_cos = MinMaxScaler().fit_transform(M_cos)          
            
  DF = pd.DataFrame(M_cos)
  
  tmp_matrix = np.copy(DF.values)
  
  same_mean = np.sum(tmp_matrix * gold_matrix.values) / np.count_nonzero(gold_matrix == 1) 
  diff_mean = np.sum(tmp_matrix * (1 - gold_matrix.values)) / np.count_nonzero(gold_matrix == 0) 
  
  return (same_mean, diff_mean, DF, vis_data)


# In[ ]:




# In[9]:

runs = {}
topics = [10, 15, 30, 45, 50, 55, 60, 65, 80, 100, 125]
iters = [50, 75, 90, 100, 125, 150, 200]


# In[ ]:

for t in topics:
  for i in iters:
    runs[(t,i)] = testGold(t, i, train_resps, gold_matrix_train)


# In[ ]:

runs_mean_diff = np.zeros((len(topics),len(iters)))
for r in runs:
  runs_mean_diff[topics.index(r[0])][iters.index(r[1])] = runs[r][0] - runs[r][1]


# In[18]:

runs = pickle.load( open( "runs.p", "rb" ) )
runs_mean_diff = pickle.load( open( "runs_mean_diff.p", "rb" ) )


# In[21]:

# table of training data with various num of topics & iterations
pd.DataFrame(runs_mean_diff, index=topics, columns=iters)


# In[16]:

tst_same_mean, tst_diff_mean, tst_DF, tst_vis_data = testGold(65, 75, test_resps, gold_matrix_test)


# In[20]:

metrics_2 = compute_metrics(runs[(65,75)][2].values, tst_DF.values, gold_matrix_train, df_gold_train, gold_matrix_test, df_gold_test)
pretty_metrics_2 = pd.DataFrame(pd.Series(metrics_2), columns = ["Score"])
pretty_metrics_2


# In[ ]:




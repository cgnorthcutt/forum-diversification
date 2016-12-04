
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


# In[70]:

# # Automate creation of gold data and gold matrix from 5 unique forum topics.

# # Useful topics

# # 5705e19d81e07b055c000bc0 - Same scriptures for bibles? - Please offer your reflections on the fact that not all Bibles have the same contents. Do you think it is important that people belonging to the same tradition share the same scriptures? Why or why not? Please post your response and respond to the posts of at least two of your peers.
# # 5702af2c10e0fd052100099c - Christianity practiced locally - Post what you found along with brief comments about what it is and why you find it distinctive or typical of Christianity where you live. Read at least five postings of your peers in preparation for answering the next discussion question.
# # 570bf4976fcfa50548000266 - Beginnings of the gospels - Now that you’ve read the introductions to the four New Testament gospels and examined their particular similarities, differences, and emphases, consider the following questions: What do readers learn about the content of these gospels and about Jesus by reading their introductions? What expectations might readers have as they begin to read each gospel? How would you now address the issue of why the New Testament has four gospels rather than a single narrative? Please post your reflections on the discussion board and respond to at least two of your peers.
# # 5705e53810e0fd052d000c3a - Canons - Often people ask about who chose which books would be included in the Bible and why. We’ve asked you to notice that the content of Bibles are different and changing. The Protestant Luther, for example, removed books from the Catholic canon and created a smaller Protestant canon. Ethiopians have a much larger canon. And now as more early Christian books are discovered in Egypt, some are suggesting these might be added to a “new New Testament.” Please share 1-3 things about the readings on canon and new discoveries that you found especially interesting or surprising. Please post your reflections and respond to the posts of at least two of your peers.
# # 570bf5946fcfa5055000026e - Nativity Scene - The nativity scene of Jesus’s birth is widely represented not only in artistic representations, but today also by small models in homes, life-size exhibits at churches, or through live staged enactments—and indeed you may have seen them in your own home, church, or another public space. If we consider these representations as interpretations of the gospel narratives, what do you think they communicate about Jesus? About the beliefs of Christians? What difference do you think it makes whether people read each gospel independently, read and compare all the gospel birth stories, or interpret Jesus’s birth through harmonized nativity scenes? Please post your reflections on the discussion board and respond to at least two of your peers. 

# topics = [
#   "5705e19d81e07b055c000bc0", 
#   "5702af2c10e0fd052100099c", 
#   "570bf4976fcfa50548000266", 
#   "5705e53810e0fd052d000c3a", 
#   "570bf5946fcfa5055000026e",
# ]

# df_gold = df[df.thread_id_from_url.isin(topics)]
# topic_map = dict(zip(topics, range(len(topics))))
# df_gold["topic_cluster"] = df_gold.thread_id_from_url.apply(lambda x: topic_map[x])

# from scipy.spatial.distance import pdist, squareform

# gold_matrix = squareform(pdist(pd.DataFrame(df_gold.topic_cluster)))
# for row in range(len(gold_matrix)):
#   for col in range(len(gold_matrix)):
#     if gold_matrix[row][col] == 0:
#       gold_matrix[row][col] = 1
#     else:
#       gold_matrix[row][col] = 0
      
# # pickle.dump( gold_matrix, open( "gold_matrix_for_HarvardX__HDS_3221_2X__1T2016.p", "wb" ) )
# # pickle.dump( df_gold, open( "gold_data_HarvardX__HDS_3221_2X__1T2016.p", "wb" ) )

# # Write to file
# cols = ['thread_id_from_url', 'body', 'num_replies', 'body_length', 'sum_body_length_per_topic', 'num_responses', 'avg_response_length', 'topic_cluster']
# pd.DataFrame(gold_matrix).to_csv("gold_matrix_for_HarvardX__HDS_3221_2X__1T2016.csv.gz", compression='gzip', index=False)
# pd.DataFrame(df_gold, columns = cols).to_csv("gold_data_HarvardX__HDS_3221_2X__1T2016.csv.gz", compression='gzip', index=False)


# In[3]:

pd.set_option('max_colwidth',5000)


# In[4]:

def remove_punctuation(text):
    return re.sub(ur"\p{P}+", "", text)


# In[16]:

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


# In[6]:

# w2v_matrix = pickle.load( open( "/Users/cgn/Dropbox (MIT)/cgn/bq/w2v_matrix.p", "rb" ) )
# vocab = np.array(pickle.load( open( "/Users/cgn/Dropbox (MIT)/cgn/bq/word.p", "rb" ) ) )
w2v_matrix = pickle.load( open( "w2v_matrix_for_HarvardX__HDS_3221_2X__1T2016.p", "rb" ) )
vocab = np.array(pickle.load( open( "vocab_for_HarvardX__HDS_3221_2X__1T2016.p", "rb" ) ) )


# In[4]:

np.shape(w2v_matrix)


# In[8]:

df = pd.read_csv('HarvardX__HDS_3221_2X__1T2016_scored_forum_responses_and_features.csv.gz', compression='gzip')
topic_id = '5705e20881e07b74f500019d'
X = df[df.thread_id_from_url==topic_id].body.values


# In[9]:

# Creation of reduced w2v_matrix model with 1-gram through 6-gram

# all_txt = " ".join(list(df.body.apply(lambda x: unicode(x, 'utf-8'))))
# translate_table = dict((ord(char), None) for char in string.punctuation)   
# all_words = all_txt.translate(translate_table).split()

# ngram_start = 1
# ngram_end = 6
# all_words_ngram = ["_".join(all_words[i:i + ngram]) for ngram in range(ngram_start, ngram_end +1) for i in range(len(all_words) - (ngram - 1))]
# unique_words_list = list(set(all_words_ngram))

# # Build hash map from word to index
# vocab_map = dict(zip(vocab, range(len(vocab))))

# # Get indices of items in vocab that are contained in unique_words_list
# idx_with_Nones = np.array([vocab_map.get(word) for word in unique_words_list])
# idx = [i for i in idx_with_Nones if i is not None]

# # Reduce w2v_matrix size by limiting to all unique words in this course.
# vocab = vocab[idx]
# w2v_matrix = w2v_matrix[idx]

# pickle.dump( w2v_matrix, open( "w2v_matrix_for_HarvardX__HDS_3221_2X__1T2016.p", "wb" ) )
# pickle.dump( vocab, open( "vocab_for_HarvardX__HDS_3221_2X__1T2016.p", "wb" ) )


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




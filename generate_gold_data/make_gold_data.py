
# coding: utf-8

# In[1]:

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function


# In[ ]:

import numpy as np
import pandas as pd
import pickle
from scipy.spatial.distance import pdist, squareform


# In[ ]:

# Automate creation of training gold data and gold matrix from 5 unique forum topics.

# Useful topics

# 5705e19d81e07b055c000bc0 - Same scriptures for bibles? - Please offer your reflections on the fact that not all Bibles have the same contents. Do you think it is important that people belonging to the same tradition share the same scriptures? Why or why not? Please post your response and respond to the posts of at least two of your peers.
# 5702af2c10e0fd052100099c - Christianity practiced locally - Post what you found along with brief comments about what it is and why you find it distinctive or typical of Christianity where you live. Read at least five postings of your peers in preparation for answering the next discussion question.
# 570bf4976fcfa50548000266 - Beginnings of the gospels - Now that you’ve read the introductions to the four New Testament gospels and examined their particular similarities, differences, and emphases, consider the following questions: What do readers learn about the content of these gospels and about Jesus by reading their introductions? What expectations might readers have as they begin to read each gospel? How would you now address the issue of why the New Testament has four gospels rather than a single narrative? Please post your reflections on the discussion board and respond to at least two of your peers.
# 5705e53810e0fd052d000c3a - Canons - Often people ask about who chose which books would be included in the Bible and why. We’ve asked you to notice that the content of Bibles are different and changing. The Protestant Luther, for example, removed books from the Catholic canon and created a smaller Protestant canon. Ethiopians have a much larger canon. And now as more early Christian books are discovered in Egypt, some are suggesting these might be added to a “new New Testament.” Please share 1-3 things about the readings on canon and new discoveries that you found especially interesting or surprising. Please post your reflections and respond to the posts of at least two of your peers.
# 570bf5946fcfa5055000026e - Nativity Scene - The nativity scene of Jesus’s birth is widely represented not only in artistic representations, but today also by small models in homes, life-size exhibits at churches, or through live staged enactments—and indeed you may have seen them in your own home, church, or another public space. If we consider these representations as interpretations of the gospel narratives, what do you think they communicate about Jesus? About the beliefs of Christians? What difference do you think it makes whether people read each gospel independently, read and compare all the gospel birth stories, or interpret Jesus’s birth through harmonized nativity scenes? Please post your reflections on the discussion board and respond to at least two of your peers. 

topics = [
  "5705e19d81e07b055c000bc0", 
  "5702af2c10e0fd052100099c", 
  "570bf4976fcfa50548000266", 
  "5705e53810e0fd052d000c3a", 
  "570bf5946fcfa5055000026e",
]

df_gold = df[df.thread_id_from_url.isin(topics)]
topic_map = dict(zip(topics, range(len(topics))))
df_gold["topic_cluster"] = df_gold.thread_id_from_url.apply(lambda x: topic_map[x])

gold_matrix = squareform(pdist(pd.DataFrame(df_gold.topic_cluster)))
for row in range(len(gold_matrix)):
  for col in range(len(gold_matrix)):
    if gold_matrix[row][col] == 0:
      gold_matrix[row][col] = 1
    else:
      gold_matrix[row][col] = 0
      
# pickle.dump( gold_matrix, open( "gold_matrix_for_HarvardX__HDS_3221_2X__1T2016.p", "wb" ) )
# pickle.dump( df_gold, open( "gold_data_HarvardX__HDS_3221_2X__1T2016.p", "wb" ) )

# Write to file
cols = ['thread_id_from_url', 'body', 'num_replies', 'body_length', 'sum_body_length_per_topic', 'num_responses', 'avg_response_length', 'topic_cluster']
pd.DataFrame(gold_matrix).to_csv("gold_matrix_train_HarvardX__HDS_3221_2X__1T2016.csv.gz", compression='gzip', index=False)
pd.DataFrame(df_gold, columns = cols).to_csv("gold_data_train_HarvardX__HDS_3221_2X__1T2016.csv.gz", compression='gzip', index=False)


# In[ ]:

# Automate creation of testing gold data and gold matrix from 5 unique forum topics.

topics = [
  "5717e338c997f905a30003b9",
  "5705e2e310e0fd1536000501",
  "571fe921c83713056c00033a",
  "5705e41281e07b0517000c22",
  "5702ad8b81e07b05170009fe",
  "570e668f6fcfa5053c0003cd",
]

df_gold = df[df.thread_id_from_url.isin(topics)]
topic_map = dict(zip(topics, range(len(topics))))
df_gold["topic_cluster"] = df_gold.thread_id_from_url.apply(lambda x: topic_map[x])

gold_matrix = squareform(pdist(pd.DataFrame(df_gold.topic_cluster)))
for row in range(len(gold_matrix)):
  for col in range(len(gold_matrix)):
    if gold_matrix[row][col] == 0:
      gold_matrix[row][col] = 1
    else:
      gold_matrix[row][col] = 0
      
# pickle.dump( gold_matrix, open( "gold_matrix_for_HarvardX__HDS_3221_2X__1T2016.p", "wb" ) )
# pickle.dump( df_gold, open( "gold_data_HarvardX__HDS_3221_2X__1T2016.p", "wb" ) )

# Write to file
cols = ['thread_id_from_url', 'body', 'num_replies', 'body_length', 'sum_body_length_per_topic', 'num_responses', 'avg_response_length', 'topic_cluster']
pd.DataFrame(gold_matrix).to_csv("gold_matrix_test_HarvardX__HDS_3221_2X__1T2016.csv.gz", compression='gzip', index=False)
pd.DataFrame(df_gold, columns = cols).to_csv("gold_data_test_HarvardX__HDS_3221_2X__1T2016.csv.gz", compression='gzip', index=False)


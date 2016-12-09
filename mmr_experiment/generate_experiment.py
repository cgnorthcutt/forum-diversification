
# coding: utf-8

# In[15]:

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function


# In[16]:

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import mmr
reload(mmr)


# In[17]:

# BigQuery code used to select the top 10 forum discussions
# which have the highest sum of scores across all comments
# in the forum discussion.

# SELECT mongoid as thread_id_from_url, title, body as question
# FROM [harvardx-data:HarvardX__HDS3221_2x__1T2016_latest.forum] A
# JOIN 
# (
#   SELECT
#     thread_id_from_url, sum(score) as sum_score, nscored_comments
#   FROM
#     (
#     SELECT 
#       comment_thread_id as thread_id_from_url, 
#       child_count as score, 
#       SUM(child_count > 0) OVER (PARTITION BY comment_thread_id) as nscored_comments,
#       COUNT(*) OVER (PARTITION BY comment_thread_id) as ncomments
#     FROM [harvardx-data:HarvardX__HDS3221_2x__1T2016_latest.forum]
#     where parent_id IS NULL # Not a reply to a comment, just a comment.
#     and comment_thread_id IS NOT NULL
#   )
#   WHERE nscored_comments >= 20
#   #AND ncomments >= 20
#   GROUP BY thread_id_from_url, nscored_comments
# ) B
# ON A.mongoid = B.thread_id_from_url
# where _type = "CommentThread"
# and not title contains "Introduce Yourself"
# and not title contains "Nondiscrimination/anti-harassment statement"
# and not title contains "Farewell"
# and not title contains "Help Thread"
# ORDER BY nscored_comments DESC

# -- SELECT
# --   *,
# --   sum_body_length_per_topic / num_responses AS avg_response_length
# -- FROM
# -- (
# --   SELECT 
# --     comment_thread_id as thread_id_from_url, 
# --     body, 
# --     child_count as num_replies, 
# --     LENGTH(body) as body_length,
# --     SUM(LENGTH(body)) OVER (PARTITION BY comment_thread_id) as sum_body_length_per_topic,
# --     COUNT(*) OVER (PARTITION BY comment_thread_id) as num_responses
# --   FROM [harvardx-data:HarvardX__HDS3221_2x__1T2016_latest.forum]
# --   #where comment_thread_id ="5705e2fc10e0fd054b000bce"
# --   where parent_id IS NULL
# --   and comment_thread_id IS NOT NULL
# -- )
# -- where num_responses >= 5
# -- order by avg_response_length, thread_id_from_url, num_replies desc


# In[18]:

def get_test_comments(all_comments, all_scores, exclude_set, num_output): 
  # exclude_set: don't output a comment that we'll be comparing with
  # set all_scores of exclude_set comments to 0 -> will make prob.
  exclude_indices = [list(all_comments).index(cmt) for cmt in exclude_set]
  
  for idx in exclude_indices:
    all_scores[idx] = 0
  
  probabilities = all_scores/float(sum(all_scores))
  
  comment_idxs = np.random.choice(len(all_comments), num_output, p=probabilities)
  return [all_comments[cmt_idx] for cmt_idx in comment_idxs]


# In[19]:

forum_topics = pd.read_csv("mmr_experiment_discussion_forum_topics.csv").T.to_dict().values()
df = pd.read_csv('../data/HarvardX__HDS_3221_2X__1T2016_scored_forum_responses_and_features.csv.gz', compression='gzip')


# In[20]:

K = 5
lambdas = [0.25, 0.75]
ntrials = 5


# In[26]:

experiment_observed = []
experiment_truth = []

for i, topic_info in enumerate(forum_topics):
  for lam in lambdas:
    thread_id = df["thread_id_from_url"]
    topic_forum = df[df["thread_id_from_url"] == topic_info["thread_id_from_url"]]
    all_comments = topic_forum.body.values
    all_scores = MinMaxScaler().fit_transform(topic_forum.num_replies.values.reshape((len(topic_forum),1)))
    all_scores = np.array([score[0] for score in all_scores])
    
    pairwise = mmr.tfidf_pca(all_comments)
    
    # Baseline uses lam = 1.0
    comment_indices_baseline = mmr.mmr(pairwise, all_scores, K, lam=1.0)
    baseline_ranking = all_comments[comment_indices_baseline]
    
    # Run mmr with PCA tfidf similarity matrix
    comment_indices = mmr.mmr(pairwise, all_scores, K, lam)
    mmr_ranking = all_comments[comment_indices]
    
    # Rankings are the same, nothing to compare for experiment.
    # Remove and continue
    if set(comment_indices_baseline) == set(comment_indices):
      # if lam = 0.75 then 0.25 was added and needs to be removed
      if lam == 0.75:
        experiment_observed = experiment_observed[:-5]
        experiment_truth = experiment_truth[:-5]
      break
    
    both_rankings = [baseline_ranking, mmr_ranking]
    # print(i, lam, comment_indices)
    
    exclude_set = set(list(baseline_ranking) + list(mmr_ranking))
    test_comments = get_test_comments(all_comments, all_scores, exclude_set, ntrials)
    
    for test_comment in test_comments:
      # Generate randomly a 0 or 1.
      choice = np.random.randint(2)
      
      experiment_observed.append({
          "title":topic_info["title"],
          "question":topic_info["question"],
          "A":both_rankings[choice],
          "B":both_rankings[1 - choice],
          "C":test_comment,
        })
      
      experiment_truth.append({
          "A": "baseline" if choice == 0 else "mmr",
          "B": "mmr" if choice == 0 else "baseline",
          "mmr_lambda": lam,
        })


# In[29]:

# Fetch first 100 trials for evaluation.
experiment_truth = np.array(experiment_truth[:100])
experiment_observed = np.array(experiment_observed[:100])


# In[14]:

# pd.DataFrame(list(experiment_truth))


# In[15]:

for person in ["curtis.txt", "carson.txt", "kim.txt", "naichun.txt"]:
  idx = range(100)
  np.random.shuffle(idx)
  pickle.dump(experiment_truth[idx], open("truth_"+person+".p", 'wb'))
  
  f = open(person, 'wb')
  for trial_num, exp_trial in enumerate(experiment_observed[idx]):
    print("Trial", trial_num, file=f)
    print(file=f)
    print("|------------|", file=f)
    print("| Question Q |", file=f)
    print("|------------|", file=f)
    print(repr(exp_trial["question"]), file=f)
    print(file=f)

    for lst in ["A", "B"]:
      print("|--------|", file=f)
      print("| List", lst,"|", file=f)
      print("|--------|", file=f)
      print(file=f)

      for i, c in enumerate(exp_trial[lst]):
        print("["+str(i+1)+"]", repr(c), file=f)
        print(file=f)

      print(file=f)

    print("|-----------|", file=f)
    print("| Comment C |", file=f)
    print("|-----------|", file=f)
    print(repr(exp_trial["C"]), file=f)
    print(file=f)
    print(file=f)
    print(file=f)


# In[13]:

pd.DataFrame(list(pd.read_pickle("truth_curtis.txt.p"))).mmr_lambda.value_counts()


# In[ ]:




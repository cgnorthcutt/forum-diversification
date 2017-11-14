# Forum Ranking Diversification

### News!

Published in Learning@Scale 2017. Paper: https://dl.acm.org/citation.cfm?id=3054016

If you find this repo or the paper helpful, please cite us:

```
@inproceedings{Northcutt:2017:CRD:3051457.3054016,
 author = {Northcutt, Curtis G. and Leon, Kimberly A. and Chen, Naichun},
 title = {Comment Ranking Diversification in Forum Discussions},
 booktitle = {Proceedings of the Fourth (2017) ACM Conference on Learning @ Scale},
 series = {L@S '17},
 year = {2017},
 isbn = {978-1-4503-4450-0},
 location = {Cambridge, Massachusetts, USA},
 pages = {327--330},
 numpages = {4},
 url = {http://doi.acm.org/10.1145/3051457.3054016},
 doi = {10.1145/3051457.3054016},
 acmid = {3054016},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {discussion forum, embeddings, information retrieval, online courses, ranking diversification, search},
} 
```

### Why diversify?

Text ranking systems (e.g. Facebook post comments, Amazon product
reviews, Reddit forums) are ubiquitous, yet many suffer from a common
problem. When items (e.g. responses or comments) are ranked primarily by
text content and rating (e.g. like/unlike, +/-,
etc.), then similar items tend to receive similar scores, often
producing redundant items with similar ranking. For example, if “Great
job!” is ranked first, then “Great job.” is likely to be ranked second.
Moreover, higher ranking items tend to only represent the majority
opinion, since there are more users in the majority group to up-vote
items sharing their sentiment; thus, for systems with thousands of items
in a single forum, since most users only view the highest-ranked items,
most users will often only be exposed to the majority opinion.

In this paper, we develop an algorithm for forum comment ranking
diversification using maximal marginal relevance (MMR) to linearly
interpolate between the original item ranking (relevance) and the
similarity of an item to higher-ranked items (diversity). A single
parameter, λ, is used to adjust this trade-off. Item similarity
is captured using the cosine similarity of tf-idf bag of words
representation, where each word is embedded using a PCA+TFIDF embedding model
trained on a corpora of 100,000+ edX course discussion forum responses.
We apply our model to the forum discussions of an online course, MITx
6.00.1x, where capturing the diversity of responses may be of high
importance to debunk misconceptions held by the majority of forum
respondents and to capture the diversity of posts across thousands of
learners. 

### Corpora

Textual forum posts and replies will be obtained via web-scraping. A
database for multiple MITx online edX courses will store the following
columns:

1.  username
2.  comment\_text
3.  comment\_type (reply or post)
4.  original comment rank.

Original comment scores (number of likes) cannot be inferred via
web-scraping. Instead we weakly estimate the score using the
current original ranking and assuming a uniform distribution of scores
from 0 to 1.

### Baseline Measure

The baseline model simply ranks comments by their number of replies and upvotes.

### Evaluation

We measure the effect on learning outcomes using a double-blind experiment 
with 300 tests across 3 reviewers. Reviewers are asked to answer 3 questions between two
rankings, one of which has secretely been ordered by our algorithm. Questions focus on
(1) diversity, (2) redundancy, (3) inclusion of the content of another comment in the forum.
Cohen's cappa is used to show inter-rater reliability. For all three questions,
reviewers selected our algorithm's ordering of the comments by a significant margin
for all three questions.

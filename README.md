# Forum Ranking Diversification


### Abstract

Text ranking systems (e.g. Facebook post comments, Amazon product
reviews, Reddit forums) are ubiquitous, yet many suffer from a common
problem. When items (e.g. responses or comments) are ranked primarily by
text content and rating (e.g. like/unlike, \( \uparrow \)/\( \downarrow \), +/-,
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
parameter, \( \lambda \), is used to adjust this trade-off. Item similarity
is captured using the cosine similarity of tf-idf bag of words
representation, where each word is embedded using a word2vec model
trained on a corpora of 100,000+ edX course discussion forum responses.
We apply our model to the forum discussions of an online course, MITx
6.00.1x, where capturing the diversity of responses may be of high
importance to debunk misconceptions held by the majority of forum
respondents and to capture the diversity of posts across thousands of
learners. Similarly to other diversification methods, we evaluate our
model using an Amazon Mechanical Turk user study comparing our ranking
and a baseline diversified ranking versus the original ranking.

### Corpora

Textual forum posts and replies will be obtained via web-scraping. A
database for multiple MITx online edX courses will store the following
columns:

1.  username

2.  comment\_text

3.  comment\_type (reply or post)

4.  original comment rank.

Original comment scores (number of likes) cannot be inferred via
web-scraping. Instead we will weakly estimate the score using the
current original ranking and assuming a uniform distribution of scores
from 0 to 1.

### Baseline Measure

The baseline model is the same as the proposed model, except for the
computation of item similarity. Instead of a computing the cosine
similarity of word2vec bag of words tf-idf representations, we use the
ratio of words in common to unique words as the similarity score. This
baseline approach does not take into account the semantic space of the
text, only the individual token components. The power of our proposed
method is that it captures the similarity of “He was once very strong”
and “The man used to have strength”, whereas the baseline model does
not.

### Evaluation

Given the time limitations of this project, we are unable to implement
forum ranking diversification in a live course and measure the effect on
learning outcomes using an AB test experiment. Instead, we use a
Mechanical Turk experiment where half the users are given the first five
items of the baseline ranking versus the original ranking and asked
“Which list of five items contains more diverse responses (differs more
in content, ideas, and/or meaning)?” We then ask the same question for
the first five items of our model’s ranking versus the original ranking.
As the final evaluation, we compare the improvement of our model versus
the improvement of the baseline model, for a set of forum posts, using a
matched pairs t-test.

Whether to include known cases (to verify Mechanical Turk users are
answering correctly), the number of users who see each ranking, the
number of forum posts to consider, and whether to look at post
diversification or reply diversification are all factors which are still
to be determined.

### Report Link

The Google Doc containing updates can be found
[here](https://docs.google.com/document/d/15mwOCX2Sg1KTpXNSMEgPv9FruaQfDvRVV0C9D9SOqhk/edit).

### Github Link

Our Github link can be found
[here](https://github.mit.edu/cgn/forum-diversification).


# coding: utf-8

# In[1]:

import pandas as pd


# In[6]:

print pd.read_csv("model_evaluation.csv").drop("Random Forest Accuracy", axis = 1).to_latex(index=False)


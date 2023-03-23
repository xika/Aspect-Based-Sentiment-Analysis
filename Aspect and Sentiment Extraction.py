#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


from pyabsa import ATEPCCheckpointManager
from pyabsa import available_checkpoints
import re 
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings("ignore")


# In[3]:


data = pd.read_csv("input/amazon_alexa.tsv",sep='\t')


# In[4]:


data['rating'] =  data['rating'].apply(lambda x : 1 if x > 3 else 0)
data = data.drop(['date','variation','feedback'],axis=1)


# In[5]:


data['rating'].value_counts()


# In[6]:


STOPWORDS = stopwords.words("english")


# In[7]:


def clean(x):
    x =  x.lower()
    x  =  re.sub("[^\w\d]"," ",x)
    x = " ".join(t for t in x.split() if t not in STOPWORDS)
    return x


# In[8]:


data['verified_reviews'] = data['verified_reviews'].apply(lambda x : clean(x))


# In[ ]:


text =list(data['verified_reviews'][10:15].values) 


# In[ ]:


checkpoint_map = available_checkpoints()


# In[9]:


aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint='english',
                                   auto_device=True  
                                   )


# In[ ]:


atepc_result = aspect_extractor.extract_aspect(inference_source=text,  #
                          pred_sentiment=True,  
                          )


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





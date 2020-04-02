#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import s3fs
from smart_open import open
import boto3
from io import StringIO # python3; python2: BytesIO 
from boto3.s3.transfer import TransferConfig
import metrics
import torch
from transformers import *
import numpy as np


# In[ ]:


column_of_interest = ["text_ tokens"]
train_set = pd.read_csv('s3://recsys-challenge-2020/train_set.csv', encoding="utf-8",
                     usecols= [1])


# In[ ]:


column_of_interest = ["text_ tokens", "engaging_user_id"]
train_set = pd.read_csv('s3://recsys-challenge-2020/train_set.csv', encoding="utf-8",
                     usecols= [1, 4])


# In[ ]:


train_set.head()


# In[ ]:


len(train_set)


# In[ ]:


tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
model = BertModel.from_pretrained('/dev/bert/')


# In[ ]:


iterator = 0
for chunk in np.array_split(train_set, 1000000):
    print(iterator)
    iterator = iterator + 1
    df_embeddings = pd.DataFrame()
    df_embeddings = chunk[["engaging_user_id"]]
    df_embeddings['text_embeddings'] = chunk['text_ tokens'].apply(lambda x : model(torch.tensor(list(map(int, x.split('\t')))).unsqueeze(0))[0][0][0])
    df_embeddings.to_csv('s3://recsys-challenge-2020/embeddings_user.csv', mode='a', header=False)

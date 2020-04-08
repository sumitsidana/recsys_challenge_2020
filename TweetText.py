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
import ast
import time


# In[ ]:


column_of_interest = ["text_ tokens"]
train_set = pd.read_csv('s3://recsys-challenge-2020/train_set.csv', encoding="utf-8",
                     usecols= [1])
val_set = pd.read_csv('s3://recsys-challenge-2020/val_set.csv', encoding="utf-8",
                     usecols= [1])
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)


# In[ ]:


train_set.head()


# In[ ]:


def calculate_text(row):
    tweet_tokens = tokenizer.decode(list(map(int, row.split('\t'))))
    return tweet_tokens


# In[ ]:


train_set['user_text'] = train_set['text_ tokens'].apply(lambda x: calculate_text(x))


# In[ ]:


val_set['user_text'] = val_set['text_ tokens'].apply(lambda x: calculate_text(x))


# In[ ]:


train_set.to_csv('s3://recsys-challenge-2020/train_set_text.csv', index = False)


# In[ ]:


val_set.to_csv('s3://recsys-challenge-2020/val_set_text.csv', index = False)


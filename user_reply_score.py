#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
model = BertModel.from_pretrained('/dev/bert/')


# In[4]:


user_tokens = pd.read_csv('s3://recsys-challenge-2020/user_tokens.csv')


# In[9]:


column_of_interest = ["engaging_user_id", "text_ tokens"]
val_set = pd.read_csv('s3://recsys-challenge-2020/val_set_reply.csv', encoding="utf-8",
                     usecols= [1, 4])


# In[11]:


user_tokens_val_set = pd.merge(val_set, user_tokens, how = 'left', left_on = 'engaging_user_id', right_on = 'engaging_user_id', sort=False)
user_tokens_val_set.columns = [c.replace(' ', '_') for c in user_tokens_val_set.columns]


# In[59]:


def calculate_average(row1, row2):
    if pd.isna(row1):
        return 0.5
    sum_tensors = torch.zeros([768], dtype=torch.float32)
    # row1 can be nan, as, there are cold users in validation set.
    tweet_token_list = ast.literal_eval(row1)
    for token_list in tweet_token_list:
        token_list_embeddings = model(torch.tensor(list(map(int, token_list.split('\t')))).unsqueeze(0))[0][0][0]
        sum_tensors = sum_tensors + token_list_embeddings
    avg = sum_tensors/len(row1)
    tweet_average_embedding = model(torch.tensor(list(map(int, row2.split('\t')))).unsqueeze(0))[0][0][0]
    score = torch.dot(avg, tweet_average_embedding)
    return score.detach().numpy().item(0)


# In[21]:


user_val_set_reply_score = pd.DataFrame()


# In[ ]:


user_val_set_reply_score['reply_score'] = user_tokens_val_set.apply (lambda z: calculate_average(z.text__tokens_y, z.text__tokens_x), axis = 1)


# In[62]:


user_val_set_reply_score.to_csv('s3://recsyschallenge2020/user_val_set_reply_score', index = False)


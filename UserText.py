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


user_tokens = pd.read_csv('s3://recsys-challenge-2020/user_tokens.csv')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
model = BertModel.from_pretrained('/dev/bert/')


# In[ ]:


def calculate_text(row):
    tweet_token_list = ast.literal_eval(row)
    sum_text = ''
    for token_list in tweet_token_list:
        token_list_text = tokenizer.decode(list(map(int, token_list.split('\t'))))
        sum_text = sum_text + token_list_text+ ' '
    return sum_text


# In[ ]:


user_tokens['user_text'] = user_tokens['text_ tokens'].apply(lambda x: calculate_text(x))


# In[ ]:


user_tokens.to_csv('s3://recsys-challenge-2020/user_tokens_text.csv', index = False)


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import s3fs
import boto3
from io import StringIO # python3; python2: BytesIO 
from boto3.s3.transfer import TransferConfig
import torch
from transformers import *
import numpy as np
import ast
import time
import timeit


# In[ ]:


tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
model = BertModel.from_pretrained('bert-base-multilingual-cased').cuda()


# In[ ]:


user_tokens = pd.read_csv('s3://recsys-challenge-2020/user_tokens.csv')
column_of_interest = ["engaging_user_id", "text_ tokens"]
val_set = pd.read_csv('s3://recsys-challenge-2020/val_set_reply.csv', encoding="utf-8",
                     usecols= [1, 4])
user_tokens_val_set = pd.merge(val_set, user_tokens, how = 'left', left_on = 'engaging_user_id', right_on = 'engaging_user_id', sort=False)
user_tokens_val_set.columns = [c.replace(' ', '_') for c in user_tokens_val_set.columns]


# In[ ]:


def calculate_average_gpu(row1, row2):
    if pd.isna(row1):
        return 0.5
    sum_tensors = torch.zeros([768], dtype=torch.float32).cuda()
    tweet_token_list = ast.literal_eval(row1)
    len_row1 = len(row1)
    for token_list in tweet_token_list:
        list_of_tokens = list(map(int, token_list.split('\t')))
        
        tensor_tokens = torch.tensor(list_of_tokens).cuda()
        tensor_tokens_unsqueeze = tensor_tokens.unsqueeze(0).cuda()
        model_tokens  = model(tensor_tokens_unsqueeze)
        
        model_tokens_1d = model_tokens[0].cuda()
        model_tokens_2d = model_tokens_1d[0].cuda()
        token_list_embeddings =  model_tokens_2d[0].cuda()
        sum_tensors = (sum_tensors + token_list_embeddings).cuda()
    avg = (sum_tensors/len_row1).cuda()
    
    
    list_of_tokens_val = list(map(int, row2.split('\t')))
    tensor_tokens_val = torch.tensor(list_of_tokens_val).cuda()
    tensor_tokens_unsqueeze_val = tensor_tokens_val.unsqueeze(0).cuda()
    model_tokens_val = model(tensor_tokens_unsqueeze_val)
    model_tokens_1d_val = model_tokens_val[0].cuda()
    model_tokens_2d_val = model_tokens_1d_val[0].cuda()
    tweet_average_embedding = model_tokens_2d_val[0].cuda()
    
    score = torch.dot(avg, tweet_average_embedding).cuda()
    return score.cpu().detach().numpy().item(0)


# In[ ]:


user_val_set_reply_score = pd.DataFrame()
user_val_set_reply_score['reply_score'] = user_tokens_val_set.apply (lambda z: calculate_average_gpu(z.text__tokens_y, z.text__tokens_x), axis = 1)


# In[ ]:


user_val_set_reply_score.to_csv('s3://recsyschallenge2020/user_val_set_reply_score', index = False)


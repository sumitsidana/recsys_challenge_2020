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
import metrics


# In[ ]:


pd.set_option('display.max_colwidth', -1)
model = BertModel.from_pretrained('/dev/bert/')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)


# In[ ]:


user_tokens = pd.read_csv('s3://recsys-challenge-2020/user_tokens_reply.csv')
column_of_interest = ["text_ tokens", "engaging_user_id"]
val_set = pd.read_csv('s3://recsys-challenge-2020/val.tsv', encoding="utf-8",
                     usecols= [0, 14], names=column_of_interest, sep="\x01")
print('loaded datasets')


# In[ ]:


user_tokens_val_set = pd.merge(val_set, user_tokens, how = 'left', left_on = 'engaging_user_id', right_on = 'engaging_user_id', sort=False)
user_tokens_val_set.columns = [c.replace(' ', '_') for c in user_tokens_val_set.columns]
print('join completed')
print('number of rows for which score needs to be computed: ' + str(len(user_tokens_val_set)))


# In[ ]:


def calculate_average(row1, row2, index, time1):
    if index % 1000 == 0:
        print(index)
        print(time.time() - time1)
    if pd.isna(row1):
        return 0.028
    sum_tensors = torch.zeros([768], dtype=torch.float32)
    tweet_token_list = ast.literal_eval(row1)
    for token_list in tweet_token_list:
        list_of_tokens = list(map(int, token_list.split('\t')))
        if len(list_of_tokens) > 512:
            list_of_tokens = list_of_tokens[:511]
        token_list_embeddings = model(torch.tensor(list_of_tokens).unsqueeze(0))[0][0][0]
        sum_tensors = sum_tensors + token_list_embeddings
        
    avg = sum_tensors/len(row1)
    p_user_avg_embedding = avg / torch.norm(avg)
    
    tweet_list_of_tokens = list(map(int, row2.split('\t')))
    if len(tweet_list_of_tokens) > 512:
        tweet_list_of_tokens = tweet_list_of_tokens[:511]
    tweet_embedding = model(torch.tensor(tweet_list_of_tokens).unsqueeze(0))[0][0][0]
    tweet_average_embedding = tweet_embedding / torch.norm(tweet_embedding)
    
    P_B_given_A = torch.dot(p_user_avg_embedding, tweet_average_embedding)
    
#     posterior = (likelihood*prior)  /  ((likelihood*prior) + ((1-likelihood)*(1-prior)))
    
#     P(A|B) = P(B|A) * P(A) / P(B)

#     P(B) = P(B/A)*P(A) + P(B/~A)* P(~A)

    num = P_B_given_A * 0.028
    
    
    unlikelihood = 1.0 - P_B_given_A # P(B/~A)
    anti_score = unlikelihood * 0.972
    normalizing_factor = num + anti_score # P(B)

    
    posterior = num / normalizing_factor
    
    return posterior.detach().numpy().item(0)


# In[ ]:


user_val_set_reply_score = pd.DataFrame()


# In[ ]:


time1 = time.time()
user_val_set_reply_score['reply_score'] = user_tokens_val_set.apply (lambda z: calculate_average(z.text__tokens_y, z.text__tokens_x, z.name, time1), axis = 1)
time2 = time.time()
print(time2 - time1)


# In[ ]:


user_val_set_reply_score.to_csv('s3://recsys-challenge-2020/user_reply_score.csv', index = False)


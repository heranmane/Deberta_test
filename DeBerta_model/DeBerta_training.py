#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import os
import sys
import transformers 
import torch
import torch.nn.functional as F
import dynalab
from dynalab.handler.base_handler import BaseDynaHandler, ROOTPATH
from dynalab.tasks.hs import TaskIO
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



# In[2]:


# NOTE: use the following line to import modules from your repo
sys.path.append(ROOTPATH)
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer
)


# In[3]:


# ??AutoModelForSequenceClassification.from_pretrained


# In[4]:


# config = AutoConfig.from_pretrained('.')
model = AutoModelForSequenceClassification.from_pretrained('.')
tokenizer = AutoTokenizer.from_pretrained('.')

model.to("cpu")
# model.eval()
# model


# In[114]:


# model.state_dict()


# In[5]:


get_ipython().run_line_magic('pinfo2', 'DebertaTokenizer')


# In[6]:


# read in data
df = pd.read_csv("data/training_topic_modv2.csv", error_bad_lines=False)
# df.shape
# sample of data 
# sample_df = df.sample(n=5000)
# sample_df.head()
df


# In[7]:


sample_df = df[['cleaned_tweet', 'racist']]
sample_df['racist']= sample_df['racist'].fillna(0)
# s_df = sample_df.sample(n=1000)
sample_df.shape


# In[51]:


# hate_df = sample_df[sample_df['racist']==1][:100]


# In[52]:


# print(hate_df['cleaned_tweet'])


# In[53]:


# no_hate_df = sample_df[sample_df['racist']==0][:100]


# In[54]:


# len(hate_df), len(no_hate_df)


# In[8]:


# merged_df = pd.concat([hate_df,no_hate_df])
# merged_df


# In[9]:


train_size = 0.8
train_dataset=sample_df.sample(frac=train_size,random_state=100).reset_index(drop=True)
test_dataset=sample_df.drop(train_dataset.index).reset_index(drop=True)


# In[10]:


print("FULL Dataset: {}".format(sample_df.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))


# In[73]:


torch.__version__


# In[22]:


raicis_df = (train_dataset[train_dataset['racist']==0].count())
raicis_df


# # Re-training the Model

# ### Get Data

# In[26]:


all_data = sample_df;


# In[27]:


texts = sample_df['cleaned_tweet'].values
annotations = sample_df['racist'].astype(int).values


# ### Tokenize & Pipeline

# In[28]:


batch_dummy = [tokenizer(i) for i in texts]


# In[29]:


# batch_dummy


# In[30]:


# lengths of tokens
a = [len(i.input_ids) for i in batch_dummy]
min(a), max(a)


# In[31]:


# no. of words in tweets
b = [len(i.split()) for i in texts]
min(b), max(b)


# In[32]:


batch = [tokenizer(i, padding='max_length', truncation=True, max_length=max(a)) for i in texts]
# batch


# In[33]:


min([len(i.input_ids) for i in batch])


# In[34]:


lbl = []
for l in annotations:
    if l==1:
        lbl.append([1,0])
    else: 
        lbl.append([0,1])
    
        


# In[35]:


# set(set(labels))


# In[36]:


# # 
labels = torch.tensor(lbl)
# labels

# add logic if 1 => racist, 0 = non racist 


# In[37]:


mask = torch.tensor([x.attention_mask for x in batch])


# In[38]:


# # make copy of labels tensor, this will be input_ids
# input_ids = labels.detach().clone()


# # create random array of floats with equal dims to input_ids
# rand = torch.rand(input_ids.shape)

# # mask random 15% where token is not 0 [PAD], 1 [CLS], or 2 [SEP]
# mask_arr = (rand < .15) * (input_ids != 0) * (input_ids != 1) * (input_ids != 2)

# # loop through each row in input_ids tensor (cannot do in parallel)
# for i in range(input_ids.shape[0]):
#     # get indices of mask positions from mask array
#     selection = torch.flatten(mask_arr[i].nonzero()).tolist()
#     # mask input_ids
#     input_ids[i, selection] = 3  # our custom [MASK] token == 3


# ### DataLoader

# In[39]:


input_ids = []
for i in batch:
    input_i = i.input_ids
    input_ids.append(input_i)
    


# In[40]:


print(len(input_ids))


# In[41]:


encodings = {'input_ids': input_ids , 'attention_masks': mask, 'labels': labels}


# In[42]:


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        # store encodings internally
        self.encodings = encodings

    def __len__(self):
        # return the number of samples
#         for i in batch:
#             input_ids = batch['input_ids']
        return torch.tensor(self.encodings['input_ids']) .shape[0]

    def __getitem__(self, i):
        # return dictionary of input_ids, attention_mask, and labels for index i
        return {key: torch.tensor(tensor[i],dtype=torch.int64) for key, tensor in self.encodings.items()}


# In[43]:


dataset = Dataset(encodings)


# In[44]:


loader = torch.utils.data.DataLoader(dataset, batch_size=len(input_ids), shuffle=True)


# In[45]:


len(input_ids)


# ### Training

# In[46]:


from transformers import DebertaTokenizer, DebertaForSequenceClassification, DebertaConfig
from transformers import AdamW
from tqdm.auto import tqdm


# In[47]:


config = DebertaConfig()
#     vocab_size=50265,
#     hidden_size=768,
#     num_hidden_layers=12,
#     num_attention_heads=12,
#     intermediate_size=3072,
#     hidden_act='gelu',
#     hidden_dropout_prob=0.1,
#     attention_probs_dropout_prob=0.1,
#     max_position_embeddings=512,
#     type_vocab_size=0,
#     initializer_range=0.02,
#     layer_norm_eps=1e-07,
#     relative_attention=False,
#     max_relative_positions=-1,
#     pad_token_id=0,
#     position_biased_input=True,
#     pos_att_type=None,
#     pooler_dropout=0,
#     pooler_hidden_act='gelu',
#     **kwargs,


# In[48]:


# model = DebertaForSequenceClassification(config)


# In[53]:



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# and move our model over to the selected device
# model.to(device)


# In[50]:


# with open('pytorch_model.bin', mode='rb') as file:
#     fileContent = file.read()
#     print(fileContent)


# In[54]:


model.train(mode=True)
optim = AdamW(model.parameters(), lr=1e-4)


# In[55]:


epochs = 11

for epoch in range(epochs):
    # setup loop with TQDM and dataloader
    loop = tqdm(loader, leave=True)
    for batch in loop:
        # initialize calculated gradients (from prev step)
        optim.zero_grad()
        
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_masks'].to(device)
        labels = labels.to(device)
#         print(input_ids[0])
        # process
        outputs = model(input_ids,attention_mask=attention_mask,
                        labels=labels) 
        
#  
        # extract loss
        loss = outputs.loss
        # calculate loss for every parameter that needs grad update
        loss.backward()
        # update parameters
        optim.step()
        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())


# In[41]:


len(outputs.logits)


# In[119]:


from sklearn.metrics import precision_recall_fscore_support


# In[120]:


test = merged_df.sample(n=100)


# In[195]:


test_texts = test['cleaned_tweet']
test_texts


# In[196]:


# tokenize and output inference 
all_inference_output_tweets= []
for tweet in test_texts:
    input_tweets = tokenizer(tweet, return_tensors="pt")
    with torch.no_grad():
# #     print(input_tweets)
        inference_output_tweets = model(**input_tweets)
# #     print(inference_output_tweets)
        inference_output_tweets = F.softmax(inference_output_tweets[0]).squeeze()
        all_inference_output_tweets.append(inference_output_tweets)
    

#         print(inference_output_tweets)


# In[141]:


print(inference_output_tweets)


# In[197]:


# ID and Model Labeling 
labeled_tweets = []
for i in range(len(test)):
    response= {}
    index = int(all_inference_output_tweets[i].argmax())
    response["label"] = {0: "0",1: "1"}[index]
    response['prob'] = all_inference_output_tweets[i][index]
    response['id']= test['cleaned_tweet'].iloc[i]
    labeled_tweets.append(response)
new_df = pd.DataFrame(labeled_tweets)

response["prob"] = {"not-hateful": float(inference_output_tweets[0]), "hateful": float(inference_output_tweets[1])}


# In[198]:


new_df


# In[199]:


new_df = new_df.rename(columns={"id": "cleaned_tweet"})
new_df.shape


# In[200]:


merged_df = pd.merge(new_df,test, on = "cleaned_tweet", how = "outer")
merged_df.tail()                    


# In[201]:


annts = (test['racist'])
annts


# In[202]:


labeled_tweets


# In[203]:


y_pred  = [a_dict['label'] for a_dict in labeled_tweets]
y_pred = np.array(y_pred, dtype=np.int64)


# In[204]:


annts = np.array(annts, dtype=np.int64)


# In[205]:


precision_recall_fscore_support(annts, y_pred, average='macro')
precision_recall_fscore_support(annts, y_pred, average='micro')
precision_recall_fscore_support(annts, y_pred, average='weighted')


# # Evaluate Model

# In[ ]:





# In[117]:


new= "I love heran"


# In[111]:


input_tweets = tokenizer(new, return_tensors="pt")


# In[112]:


input_tweets


# In[113]:


tweets_output = model(**input_tweets)


# In[114]:


tweets_output


# In[115]:





# In[116]:


response = {}
index = int(inference_output_tweets.argmax())
response["label"] = {0: "not-hateful",1: "hateful"}[index]
response['prob'] = inference_output_tweets[index]
response['id']= new
# labeled_tweets.append (response)
# new_df = pd.DataFrame(response, index=1)
response


# ### Evaluate

# #  Pre-Trained Model

# In[82]:


# # tokenize and output inference 
# all_inference_output_tweets= []
# for tweet in texts:
#     input_tweets = tokenizer(tweet, return_tensors="pt")
#     with torch.no_grad():
# # #     print(input_tweets)
#         inference_output_tweets = model(**input_tweets)
# # #     print(inference_output_tweets)
#         inference_output_tweets = F.softmax(inference_output_tweets[0]).squeeze()
#         all_inference_output_tweets.append(inference_output_tweets)
    

# #         print(inference_output_tweets)
    


# In[83]:


# # ID and Model Labeling 
# labeled_tweets = []
# for i in range(len(merged_df)):
#     response= {}
#     index = int(all_inference_output_tweets[i].argmax())
#     response["label"] = {0: "not-hateful",1: "hateful"}[index]
#     response['prob'] = all_inference_output_tweets[i][index]
#     response['id']= sample_df['cleaned_tweet'].iloc[i]
#     labeled_tweets.append(response)
# new_df = pd.DataFrame(labeled_tweets)

# response["prob"] = {"not-hateful": float(inference_output_tweets[0]), "hateful": float(inference_output_tweets[1])}


# In[81]:


new_df['label'].unique()


# In[ ]:


new_df = new_df.rename(columns={"id": "cleaned_tweet"})
new_df.shape


# In[ ]:


# sample_df


# In[ ]:


# merged_df = pd.merge(new_df,sample_df, on = "cleaned_tweet", how = "outer")
# merged_df.tail()                    


# In[ ]:


# matched_hateful = merged_df[(merged_df['racist']==1) & (merged_df['label']=='hateful')]
# print(len(matched_hateful))


# In[ ]:


# umatched_hateful = merged_df[(merged_df['racist']==0) & (merged_df['label']=='hateful')]
# print(len(umatched_hateful))


# In[ ]:


# matched_not_hateful = merged_df[(merged_df['racist']==0) & (merged_df['label']=='not-hateful')]
# print(len(matched_not_hateful))


# In[ ]:


# umatched_not_hateful = merged_df[(merged_df['racist']==1) & (merged_df['label']=='not-hateful')]
# print(len(umatched_not_hateful))


# In[ ]:


# not_hateful = merged_df[(merged_df['hate_speech']==1) & (merged_df['label']=='not-hateful')]
# text = not_hateful['tweet']
# text.values


# In[ ]:


# merged_df.to_csv('merged.csv')


# In[ ]:





# In[ ]:


# import simpletransformers
# from simpletransformers.classification import (
#     ClassificationModel, ClassificationArgs
# )
# import pandas as pd
# import logging


# In[ ]:





# logging.basicConfig(level=logging.INFO)
# transformers_logger = logging.getLogger("transformers")
# transformers_logger.setLevel(logging.WARNING)

# # Preparing train data
# train_data = [
#     [
#         " I hate black people",
#         "Send Asians Back",
#         1,
#     ],
#     [
#         "I do not feel like working",
#         "Love Ethiopian Food",
#         0,
#     ],
# ]
# train_df = pd.DataFrame(train_data)
# train_df.columns = ["text_a", "text_b", "labels"]

# # Preparing eval data
# eval_data = [
#     [
#         "Black Lives Do not matter ",
#         "Build the wall",
#         1,
#     ],
#     [
#         "Merry was the king of Rohan",
#         "Legolas was taller than Gimli",
#         0,
#     ],
# ]
# eval_df = pd.DataFrame(eval_data)
# eval_df.columns = ["text_a", "text_b", "labels"]

# # Optional model configuration
# model_args = ClassificationArgs(num_train_epochs=1)

# # Create a ClassificationModel
# model = ClassificationModel("deberta", "microsoft/deberta-base", use_cuda=False)

# # Train the model
# model.train_model(train_df, output_dir='outputs' )

# # Evaluate the model
# result, model_outputs, wrong_predictions = model.eval_model(
#     eval_df
# )

# # Make predictions with the model
# predictions, raw_outputs = model.predict(
#     [
#         [
#             "Fuck covid asians",
#             "Muslims should not come",
#         ]
#     ]
# )

# # from models.multi_choice import MultiChoiceModel


# In[ ]:


# print(predictions)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





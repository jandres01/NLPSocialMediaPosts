from __future__ import unicode_literals, print_function, division
from io import open
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import random
import pickle
from torch.autograd import Variable
from numpy import array
from numpy import argmax

def prepare_sequence(seq):
    tensor = torch.FloatTensor(seq)
    tensor = tensor.view(-1,1,len(dict))
    return autograd.Variable(tensor)

#target size = output size
class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        #self.activ = nn.functional.sigmoid()
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()
    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))
    def forward(self, s_tweet):
        lstm_out, self.hidden = self.lstm(s_tweet, self.hidden)
        tag_space = nn.functional.sigmoid(self.hidden2tag(lstm_out))
        return tag_space

#create binary one hot encoding for each word in tweet of all the words in dictionary
def zerolst():
    listofzeros = [0] * len(dict)
    return listofzeros

def createOneHotEnc(post):
  tweet = []
  #add words in dict
  for word in post:
    if word in dict:
      tweet.append(dict[word]) #fit_transforming
  single_enc = []
  if len(tweet) == 0:
    return 0
  #create one hot encoding
  for i in array(tweet):
    zlst = zerolst()
    zlst[i-1] = 1
    single_enc.append(zlst)
  return single_enc

df = pd.read_csv("/data/hibbslab/jandres/torch/cleaned_twitter_Data.csv")
#df = pd.read_csv("/data/hibbslab/jandres/torch/clean_twitter_Data.csv")
df = df.sample(frac=1).reset_index()
#df = df.head(n=20)#df.head(n=1000000)

#find score df[(df.score == '0.25')]
df['score'].replace([1], 0.25, inplace=True)
df['score'].replace([2], 0.5,inplace=True)
df['score'].replace([3], 0.75,inplace=True)
df['score'].replace([4], 1,inplace=True)

#data problem
#4 - 103, 3 - 122429, 2 - 18, 1 - 239179, 0 - 613

#place words in dict & word + pos in lst
#tweetDF = df['sentimenttext']
tweetDF = df['message']
wCount = {} #word count
dict = {} #word & position
lstWords = [] #words in dict 
split_tweets = []
for tweet in tweetDF:
  tweet = tweet.split()
  split_tweets.append(tweet)
  #wCount.append(len(tweet))
  for word in tweet:
    if word not in wCount:
      wCount[word] = 1
    else:
      wCount[word] = wCount[word] + 1

#remove tf-idf
#add words that occurred more than once
for i in range(0,len(wCount)):
  word,count = wCount.items()[i] 
  if count > 1:
    lstWords.append(word)
    dict[word] = len(lstWords) -1

EMBEDDING_DIM = len(dict)#len of dict 
HIDDEN_DIM = 128

tag_to_ix = {"No": 0, "Yes": 1}
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(dict), 1)
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

tweets = split_tweets

#Add one hot encoding & word count to original dataframe
df['split_tweet'] = tweets

#Train Model
for epoch in range(10): 
  #random.shuffle(enclst) #shuffle data between epochs
  df.sample(frac=1).reset_index()
  for i in range(0,len(tweets)):
      # Step 1. Remember that Pytorch accumulates gradients.
      # We need to clear them out before each instance
      model.zero_grad()
      # Also, we need to clear out the hidden state of the LSTM,
      # detaching it from its history on the last instance.
      model.hidden = model.init_hidden()
      # Step 2. Get our inputs ready for the network, that is, turn them into
      # Variables of word indices.
      post_enc = createOneHotEnc(df.iloc[i]['split_tweet'])
      #make sure post is not empty
      if post_enc != 0:
        sentence_in = prepare_sequence(post_enc)
        targets = autograd.Variable(torch.FloatTensor([df['score'][i]]))
        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)
        # Step 4. Compute the loss, gradients, and update the parameters by
        loss = loss_function(tag_scores[len(tag_scores)-1], targets)
        print(loss)
        loss.backward()
        optimizer.step()
        print(i)

#torch.save(model, 'test_2.pt')
torch.save(model, 'clean_LSTM_model.pt')
#torch.save(model, 'clean_LSTM_model.pt')
#model = torch.load('filename.pt')

f = open('model_dict.pt', 'wb')
pickle.dump(dict, f)
f.close()


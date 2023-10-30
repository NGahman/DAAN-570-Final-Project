#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Installing necessary dependences:


# In[ ]:


#imports:
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Embedding, Dense, MultiHeadAttention, Softmax, TextVectorization, Input
from keras import Model
from keras.models import Sequential


# In[ ]:


#Preprocessing Data
filepath = os.path.dirname(os.path.dirname(os.path.abspath("."))) + "/Datasets"
filepath = re.sub(r'\\', '/', filepath)
print(filepath)
X1 = pd.read_csv(filepath + "/Tatobea Project" + '/'+ "spa" +'.txt', sep='\t', header = None)[[0,1]].rename(columns = {0:"English", 1:"Translated"})


# In[ ]:


print(X1)


# In[ ]:


X1['Translated'] = "<ES> " + X1['Translated']


# In[ ]:


print(X1)


# In[ ]:


train_x = pd.DataFrame()


# In[ ]:


for d in [x[0] for x in os.walk(filepath + "/DGT-TM")]:
    print(d)
    print(d[-6:])
    #Ignores TMX files
    if d[-1] == ")" or d[-6:] == "DGT-TM":
        continue
    #This code actually works, however due to the dataset it takes 1+ hour to run.
    #Because of this, at least for now I'll be restricting it to just the EN-ES folder.
    te = True
    if d[-5:] != "EN-ES":
        te = False
        continue
    
    fp = re.sub(r'\\', '/', d)
    E = fp.split("/")[-1].split("-")[0]
    print(E)
    T = fp.split("/")[-1].split("-")[1]
    print(T)
    print(d + '/'+ E + "-" + T +'.txt')
    
    temp = pd.read_csv(d + '/'+ E + "-" + T +'.txt', sep='\t', header = None)[[0,1]].rename(columns = {0:"English", 1:"Translated"})
    temp['Translated'] = "<" + T + "> " + temp['Translated']
    print(temp.head())
    if te:
        X1 = pd.concat([X1, temp])
    else:
        train_x = pd.concat([train_x, temp])


# In[ ]:


Y1 = pd.DataFrame()
Y1['Translated'] = X1.pop("Translated")


# In[ ]:


print(Y1)


# In[ ]:


train_x1, test_x, train_y1, test_y = train_test_split(X1, Y1, test_size=0.5, random_state=42)


# In[ ]:





# In[ ]:


try:
    train_y = train_x.pop("Translated")
except:
    #train_x doesn't have any portions, so we'll just describe it as a normal Dataframe
    train_y = pd.DataFrame()


# In[ ]:


train_x = pd.concat([train_x, train_x1])


# In[ ]:


train_y = pd.concat([train_y, train_y1])


# In[ ]:


print(train_x)
print(train_y)
print(test_x)
print(test_y)


# In[ ]:


print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)


# In[ ]:


train_x = train_x.astype(str)
test_x = test_x.astype(str)


# In[ ]:


max_padding_series = train_x['English'].str.split(" ")


# In[ ]:


max_padding_len = max_padding_series.str.len()


# In[ ]:


print(max_padding_len)


# In[ ]:


max_padding_size = max_padding_len.max()


# In[ ]:


print(max_padding_size)


# In[ ]:





# In[ ]:


#Actual Model:
#For now, let's start by creating an exact copy of the attention is all you need model.
#We can always make changes later if we want to.
i = Input(shape=(1,), dtype=tf.string)


# In[ ]:


r = open("textvectorvocap.txt", "r", encoding="utf-8")
m = r.readline().split("*")


# In[ ]:


inp = TextVectorization(output_sequence_length = max_padding_size, vocabulary = m)
#inp = TextVectorization(output_sequence_length = max_padding_size)


# In[ ]:


print(inp)


# In[ ]:


#inp.adapt(train_x)


# In[ ]:


#dir(inp)


# In[ ]:


#print(inp.get_vocabulary())


# In[ ]:





# In[ ]:


#f = open("textvectorvocap.txt", "w", encoding="utf-8")
#f.write("*".join(inp.get_vocabulary()))
#f.close()


# In[ ]:


#r = open("textvectorvocap.txt", "r", encoding="utf-8")
#m = r.readline().split("*")


# In[ ]:


#print(m == inp.get_vocabulary())


# In[ ]:





# In[ ]:


#print(inp.vocabulary_size())


# In[ ]:


#train_x_vec = inp.predict(train_x)
#test_x_vec = inp.predict(test_x)


# In[ ]:





# In[ ]:


em = Embedding(input_dim = inp.vocabulary_size(), output_dim = 512)(inp)


# In[ ]:


#This is a temporary layer only created so that I can see the model results
d = Dense(512)(em)


# In[ ]:


s = Softmax()(d)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#model = Model(inputs=i, outputs=)


# In[ ]:





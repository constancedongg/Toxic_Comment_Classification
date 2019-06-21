'''
Keras
'''

import sys, os, re, csv, codecs, numpy as np, pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
import numpy as np
import pandas as pd


file_path = 'data/'
# file_path = 'drive/My Drive/Colab_Notebooks/'
dat = pd.read_csv(file_path + 'train.csv')
dat.head(10)

X_raw = dat['comment_text']
y_raw = dat.loc[ : ,'toxic':'identity_hate']

dat_test = pd.read_csv(file_path + "test.csv")
X_test_raw = dat_test["comment_text"]

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = dat[list_classes].values

# originally there are 168900 tokens
max_words = 20000
tokenizer = Tokenizer(num_words = max_words)

tokenizer.fit_on_texts(list(X_raw))

# tokenizer.texts_to_sequences gives the np array of indices of token for each comment, each row represents indices of tokens of that comment
tokenized_train = tokenizer.texts_to_sequences(X_raw)
tokenized_test = tokenizer.texts_to_sequences(X_test_raw)

# currently no need to tune max_len
max_len = 200
X_tr = pad_sequences(tokenized_train, maxlen = max_len)
X_te = pad_sequences(tokenized_test, maxlen = max_len)

inp = Input(shape = (max_len, ))

# need to tune embed_size
embed_size = 128
x = Embedding(max_words, embed_size)(inp)


# Use embedding layer (200 , 128) as LSTM layer
# output dimension: 60; recursively run LSTM for 200 times since embedding has 200 layers
x = LSTM(60, return_sequences = True , name = 'lstm_layer')(x)

# Max pooling: down-samples features
x = GlobalMaxPool1D()(x)

x = Dropout(0.1)(x)

x = Dense(50, activation="relu")(x)

x = Dropout(0.1)(x)

x = Dense(6, activation="sigmoid")(x)

model = Model(inputs = inp, outputs = x)
model.compile(loss = 'binary_crossentropy',optimizer = 'adam', metrics = ['accuracy'])

batch_size = 32   # 32 padded sentences for each batch
epochs = 1       # entire dataset paased forward and backward twice
model.fit(X_tr,y, batch_size = batch_size, epochs = epochs, validation_split = 0.1)

prediction_1 = model.predict(X_te , batch_size = batch_size, verbose = 0, steps = None)

submission_5 = pd.DataFrame.from_dict({'id': dat_test['id']})

for i in range(len(class_names)):
    submission_5[class_names[i]] = prediction_1[:,i]

submission_5.to_csv(file_path + 'submission_5.csv', index = False)





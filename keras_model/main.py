import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk
import pickle

nltk.download('punkt')

dataset = pd.read_csv('./data/draw.train.tsv', delimiter='\t')

# make word dictionary
dictionary_word = {"TEST" : -1}
max_len = -9876
for i in range(dataset.shape[0]):
    s = dataset.iloc[i][1]
    token = word_tokenize(s)
    if max_len < len(token):
        max_len = len(token) 
    for word in token:
        try:
            if dictionary_word[word] == '1234':
                pass
        except:
            dictionary_word[word] = dictionary_word[max(dictionary_word, key=dictionary_word.get)] + 1

# setting - train data
train_x = np.empty((0, max_len), int)
for i in range(dataset.shape[0]):
    s = dataset.iloc[i][1]
    token = word_tokenize(s)
    tmp = []
    for word in token:
        tmp.append(dictionary_word[word])
    if len(tmp) != max_len:
        for _ in range(max_len - len(tmp)):
            tmp.append(0)
    
    train_x = np.append(train_x, np.array([tmp]), axis=0)

# make label dictionary
label_dict = {"TEST" : -1}
for i in range(dataset.shape[0]):
    s = dataset.iloc[i][0]
    
    try:
        if label_dict[s] == -987:
            pass
    except:
        label_dict[s] = label_dict[max(label_dict, key=label_dict.get)] + 1

# setting - label
train_y = np.array([])
for i in range(dataset.shape[0]):
    s = dataset.iloc[i][0]
    train_y = np.append(train_y, int(label_dict[s]))

train_y = train_y.astype('int32')
train_x = train_x.astype('int32')

# save dict
with open('word.pickle','wb') as f:
    pickle.dump(dictionary_word, f)
with open('label.pickle','wb') as f:
    pickle.dump(label_dict, f)

# hyperparameter
input_dim = 100
embedding_vecor_length = 256

# model init
model = Sequential()
model.add(Embedding(input_dim, embedding_vecor_length, input_length=max_len))
model.add(Bidirectional(LSTM(256, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(12, activation='softmax'))
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# train
model.fit(train_x, train_y, epochs=100, batch_size=64)

# calc acc
scores = model.evaluate(train_x, train_y, verbose=0)
print("ACC : ", (scores[1]*100))

# save model
model.save("draw_model")

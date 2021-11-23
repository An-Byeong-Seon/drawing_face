from nltk.tokenize import word_tokenize
import nltk
import pickle
import tensorflow as tf
import numpy as np
import sys

with open('./data/word.pickle', 'rb') as f:
    dictionary_word = pickle.load(f)
with open('./data/label.pickle', 'rb') as f:
    label_dict = pickle.load(f)
max_len = 16

# s = "She has no bangs"
# s = "He has long face"
# s = "She has no beard, and has a square-shaped face"
# s = "She is middle-aged white woman"
s = sys.argv[1]


test_input = np.empty((0, max_len), int)

token = word_tokenize(s)
tmp = []
for word in token:
    tmp.append(dictionary_word[word])
if len(tmp) != max_len:
    for _ in range(max_len - len(tmp)):
        tmp.append(0)
    
test_input = np.append(test_input, np.array([tmp]), axis=0)

model = tf.keras.models.load_model("./data/draw_model")
print("predict : ", [k for k, v in label_dict.items() if v == np.argmax(model.predict(test_input))])




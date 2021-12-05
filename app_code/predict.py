from nltk.tokenize import word_tokenize
import nltk
import pickle
import tensorflow as tf
import numpy as np
import sys

def predict(inputs):
    # load dict
    with open('word.pickle', 'rb') as f:
        dictionary_word = pickle.load(f)
    with open('label.pickle', 'rb') as f:
        label_dict = pickle.load(f)

    # hardcoding
    max_len = 16

    # for test
    # s = sys.argv[1]
    # s = "She has no bangs"
    # s = "He has long face"
    # s = "She has no beard, and has a square-shaped face"
    # s = "She is middle-aged white woman"
    
    # get sequence
    s = inputs

    # settings for model input
    test_input = np.empty((0, max_len), int)

    token = word_tokenize(s)
    tmp = []
    for word in token:
        tmp.append(dictionary_word[word])
    if len(tmp) != max_len:
        for _ in range(max_len - len(tmp)):
            tmp.append(0)
        
    test_input = np.append(test_input, np.array([tmp]), axis=0)

    # load model
    model = tf.keras.models.load_model("draw_model")

    # predict key
    return [k for k, v in label_dict.items() if v == np.argmax(model.predict(test_input))]
    
    # for test
    # print("predict : ", [k for k, v in label_dict.items() if v == np.argmax(model.predict(test_input))])

if __name__ == "__main__":
    key = predict("She has no bangs")[0]
    print("key : ", key)
    print("1 : ", key[:-1])
    print("2 : ", key[-1])

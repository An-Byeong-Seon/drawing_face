import argparse

import numpy as np
import torch
import torch.nn as nn
from data_loader import DataLoader # data_loader.py

WORD_VEC_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 4
BATCH_SIZE = 1

# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self,
                 input_size, # vocab_size
                 word_vec_size, # word embbeding vector 차원
                 hidden_size, # bidirectional LSRM의 hidden state & cell state의 size
                 n_classes,
                 num_layers=4, # 쌓을 레이어 개수
                 dropout_p=0.3
                 ):
        super(RNN, self).__init__()
        
        self.input_size = input_size
        self.word_vec_size = word_vec_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        
        # 입력 차원(vocab_size), 출력 차원(word_vec_size)
        self.emb =nn.Embedding(input_size, word_vec_size) # 부터!
        
        self.lstm = nn.LSTM(input_size=word_vec_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout_p,
                            batch_first=True,
                            bidirectional=True)
        self.fc  = nn.Linear(hidden_size*2, n_classes)
        # LogSoftmax + NLLLoss instead of Softmax + CrossEntropy
        self.activation = nn.LogSoftmax(dim=1) # 마지막 차원에 softmax
        
    def forward(self, x):
        # x: (batch_size, length)
        x = self.emb(x)
        
        # x: (batch_size, length, word_vec_size)
        x, _ = self.lstm(x) # x: output, _: 마지막 time step의 hidden state & cell state
        
        # x: (batch_size, length, hidden_size*2)
        # x[:,-1]: (batch_size, 1, hidden_size*2)
        out = self.activation(self.fc(x[:,-1])) # 마지막 time step
        # self.fc(x[:,-1]): (batch_size, num_classes)
        
        return out

def predict(weight_path, text, fn):

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fn = "./descriptions/"+fn+".tsv"

    f = open(fn, "w", encoding="utf-8")
    f.write("-\t"+text)
    f.close()

    
    loaders = DataLoader(
        train_fn=fn,
        batch_size=1,
        valid_ratio=.01, # val 안 나눈다. 0은 안 받으므로 0.01
        max_vocab=999999,
        min_freq=5,
    )

    vocab_size = len(text.replace(".", " ").split(" "))

    feature = weight_path.split("weight_")[1].split('.')[0]
    classes = {"h_length": 4, "h_bang": 3, "h_curl": 3, "e_shape": 4, "f_shape": 4, "sex": 2, "nose": 2}
    num_classes = classes[feature]


    model = RNN(input_size=vocab_size,
            word_vec_size=WORD_VEC_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            n_classes=num_classes).to(device)

    model = torch.load(weight_path, map_location=device)

    for _, data in enumerate(loaders.train_loader): # batch_size만큼
            texts = data.text.to(device) # (batch_size, length)
            
            # Forward prop.
            output = model(texts) # (batch_size, num_classes)
            _, output_index = torch.max(output, 1)
            prediction  = output_index[0]

            break
            
    return prediction

if __name__ == "__main__":
    print(predict('./nets/rnn_weight_sex.pkl', "He is middle aged who looks like late 50s. He has black thick hard angled eyebrows. He has deep set eyes with double eyelids. He has very dark brown color iris. He has winkle around his eyes and forehead. He has skin tones of small full lips. He has medium sized refined nose. His face shape is triangle, and he has M-shaped forehead. His skin color is golden natural. He has short gray color hair. He has gray color of the short, boxed bread beard mark. He has long and narrow ears. He looks like he's wearing a blue hat.", str(2)))
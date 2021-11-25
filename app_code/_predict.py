import torch
import torch.nn as nn
from data_loader import DataLoader # data_loader.py

from face_maker import face_maker
from datetime import datetime

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

def predict(weight_path, file_name, vocab_size):

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    loaders = DataLoader(
        train_fn=file_name,
        batch_size=1,
        valid_ratio=.01, # val 안 나눈다. 0은 안 받으므로 0.01
        max_vocab=999999,
        min_freq=5,
    )

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
            prediction  = int(output_index[0])

            break
            
    return prediction

if __name__ == "__main__":

    en_text = "He is a Western child. His skin is bright and red dots are visible on his right cheek. He has big ears. His face is round-shaped. He has square wide forehead and short dark blonde curly hair. He has double outer eyelids and thin, light brown eyebrows. And he has big round eyes with gray-green pupils. He has a broad nose tip and low, short and small nose. Nostrils are more than half visible. He is smiling broadly and appears 8 white upper and lower teeth. He has a small mouth and thin and deep red lips."

    fn = "./descriptions/"+datetime.now().strftime('%Y%m%d%H%M%S')+".tsv"

    f = open(fn, "w", encoding="utf-8")
    f.write("-\t"+en_text)
    f.close()


    vocab_size = len(en_text.replace(".", " ").split(" "))

    e_shape_pred = predict('./nets/rnn_weight_e_shape.pkl', fn, vocab_size)
    f_shape_pred = predict('./nets/rnn_weight_f_shape.pkl', fn, vocab_size)
    h_curl_pred = predict('./nets/rnn_weight_h_curl.pkl', fn, vocab_size)
    h_bang_pred = predict('./nets/rnn_weight_h_bang.pkl', fn, vocab_size)
    h_length_pred = predict('./nets/rnn_weight_h_length.pkl', fn, vocab_size)
    nose_pred = predict('./nets/rnn_weight_nose.pkl', fn, vocab_size)
    sex_pred = predict('./nets/rnn_weight_sex.pkl', fn, vocab_size)

    result = ''.join(str(_) for _ in [e_shape_pred, f_shape_pred, h_curl_pred, h_bang_pred, h_length_pred, nose_pred, sex_pred])

    print(result)
    face_maker(e_shape_pred, f_shape_pred, h_curl_pred,
                h_bang_pred, h_length_pred, nose_pred, sex_pred)
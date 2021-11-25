import argparse

import torch
import torch.nn as nn
from data_loader import DataLoader # data_loader.py

DROPOUT = 0.3
WORD_VEC_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 4
BATCH_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

def get_parser():

    parser = argparse.ArgumentParser(description="Bidirectional LSTM")

    parser.add_argument("--dropout", type=float, default=DROPOUT,
                        help="드롭아웃")
    parser.add_argument("--word-vec-size", type=int, default=WORD_VEC_SIZE,
                        help="워드 임베딩 벡터 사이즈")
    parser.add_argument("-hidden-size", type=int, default=HIDDEN_SIZE,
                        help="히든 사이즈")
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS,
                        help="레이어 수")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="배치 사이즈")
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS,
                        help="에포크 수")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="학습률")
    parser.add_argument("--train-data", type=str, default=None,
                        help="학습 데이터")
    parser.add_argument("--test-data", type=str, default=None,
                        help="테스트 데이터")

    return parser

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

def ComputeAccr(dloader, imodel):
    correct = 0
    total = 0
    
    imodel.eval() # test mode
    for data in dloader: # batch_size만큼
        texts = data.text.to(device) # (batch_size, length)
        labels = data.label.to(device) # (batch_size, num_classes)
        
        # Forward prop.
        output = imodel(texts) # (batch_size, num_classes)
        _, output_index = torch.max(output, 1) # (batch_size, 1)
        
        total += labels.size(0)
        
        correct += (output_index == labels).sum().float()
        # print("Accuracy of Test Data: {}".format(100*correct/total))
    
    imodel.train()
    return (100*correct/total).numpy() # tensor -> numpy

if __name__ == '__main__':
    
    parser = get_parser()
    args = parser.parse_args()

    feature_name = args.test_data.split('/')[-1].split('.')[0]
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loaders = DataLoader(
        train_fn=args.train_data,
        batch_size=args.batch_size,
        valid_ratio=.2, # train:val = 8:2
        max_vocab=999999, # 크게
        min_freq=5, # 문장의 최소 단어 개수
    )

    test_loaders = DataLoader(
        train_fn=args.test_data,
        batch_size=args.batch_size,
        valid_ratio=.01, # val 안 나눈다. 0은 안 받으므로 0.01
        max_vocab=999999,
        min_freq=5,
    )

    vocab_size = len(loaders.text.vocab)
    num_classes = len(loaders.label.vocab)

    model = RNN(input_size=vocab_size,
            word_vec_size=args.word_vec_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            n_classes=num_classes,
            dropout_p=args.dropout).to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train the model
    total_step = len(loaders.train_loader)
    for epoch in range(args.num_epochs):
        for i, data in enumerate(loaders.train_loader): # batch_size만큼
            texts = data.text.to(device) # (batch_size, length)
            labels = data.label.to(device) # (batch_size, num_classes)
            
            print("[%d]" %i)
            
            # Forward prop.
            output = model(texts) # (batch_size, num_classes)
            loss = loss_func(output, labels)
            
            # Backward prop. & optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 10 == 0:
                print('Epoch[{}/{}], Step [{}/{}], Loss: {:.4f}, Accr: {:.2f}'
                    .format(epoch+1, args.num_epochs, i+1, total_step,
                            loss.item(),
                            ComputeAccr(loaders.valid_loader, model)))
    netname = './nets/rnn_weight_face.pkl'
    torch.save(model, netname, )
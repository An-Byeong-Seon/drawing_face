import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, word_vec_size, hidden_size, n_classes, num_layers=4, dropout_p=0.3):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.word_vec_size = word_vec_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_classes = n_classes

        self.emb = nn.Embedding(self.input_size, self.word_vec_size)
        self.lstm = nn.LSTM(input_size=self.word_vec_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            dropout=dropout_p, batch_first=True,
                           bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, n_classes)
        self.activation = nn.LogSoftmax(dim=-1)
        
    def forward(self, x):
        x = self.emb(x)
        x, _ = self.lstm(x)
        
        out = self.activation(self.fc(x[:, -1]))
        
        return out

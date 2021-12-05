import pandas as pd
import torchtext
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
from data_loader import DataLoader
from RNN import RNN
import os

def ComputeAccr(dloader, model, testing=False):
    correct = 0
    total = 0
    
    model.eval()
    for i, data in enumerate(dloader):
        texts = data.text.to(device)
        labels = data.label.to(device)
        
        output = model(texts)
        _, output_index = torch.max(output, 1)
        
        total += labels.size(0)
        correct += (output_index == labels).sum().float()
        
        if testing:
            print("pred : {}, ground : {}".format(output_index, labels))
    model.train()
    return (100 * correct / total).cpu().numpy()


batch_size = 128
num_epochs = 100

word_vec_size = 256
dropout_p = 0.3

hidden_size = 512
num_layers = 4

learning_rate = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# load data
loaders = DataLoader(
        train_fn='./data/draw.train_real.tsv',
        batch_size = batch_size,
        valid_ratio = .1,
        device = 0,
        max_vocab = 999999,
        min_freq = 5
)

test_loaders = DataLoader(
            train_fn='./data/draw.test_real.tsv',
            batch_size = batch_size,
            valid_ratio = .01,
            device = 0,
            max_vocab = 999999,
            min_freq = 5
)

# hyperparameter
vocab_size = len(loaders.text.vocab)
num_classes = len(loaders.label.vocab)

# our model
model = RNN(vocab_size, word_vec_size, hidden_size, num_classes, num_layers, dropout_p).to(device)

loss_func = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

total_step = len(loaders.train_loader)
for epoch in range(num_epochs):
    for i, data in enumerate(loaders.train_loader):
        print(type(data))
        texts = data.text.to(device)
        labels = data.label.to(device)

        outputs = model(texts)
        loss = loss_func(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print("Epoch [{}/{}], step [{}/{}], Loss : {:.4f}, Accr: {:.2f}".format(epoch+1,num_epochs, i+1, total_step, loss.item(), ComputeAccr(loaders.valid_loader, model)))
        
# final accuracy
print("ACC : %.2f"%ComputeAccr(loaders.valid_loader, model))

# save weight
netname = './nets/draw_last.pkl'

if not os.path.exists('./nets'): 
    os.mkdir('./nets')
torch.save(model, netname)



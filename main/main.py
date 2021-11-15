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

def ComputeAccr(dloader, model):
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
    model.train()
    return (100 * correct / total).cpu().numpy()


batch_size = 128
num_epochs = 20

word_vec_size = 256
dropout_p = 0.3

hidden_size = 512
num_layers = 4

learning_rate = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


df = pd.read_csv('./data/sms.tsv', sep='\t')
#print(df.columns)
#print(df.shape)

max_length = 256

classes = sorted(set(df['label']))
class_to_idx = {}
for i, c in enumerate(classes):
    class_to_idx.update({c:i})

nclass = len(classes)

new_df = pd.DataFrame({'label' : df['label'], 
                        'sms' : df['sms'].str.slice(start=0, stop=max_length)})
new_df = pd.DataFrame(new_df.drop_duplicates())
df_shuffled = new_df.sample(frac=1).reset_index(drop=True)


train_ratio = 0.9

'''
# train dataset
s, e = 0, int(df_shuffled.shape[0] * train_ratio)
df_train = pd.DataFrame({'label' : df_shuffled['label'][s:e],
                        'sms' : df_shuffled['sms'][s:e]})

# test dataset
s, e = e, e + int(df_shuffled.shape[0] * (1.0 - train_ratio))
df_test = pd.DataFrame({'label' : df_shuffled['label'][s:e],
                        'sms' : df_shuffled['sms'][s:e]})


df_train.to_csv('./data/sms.maxlen.uniq.shuf.train.tsv', header=False, index=False, sep='\t')
df_test.to_csv('./data/sms.maxlen.uniq.shuf.test.tsv', header=False, index=False, sep='\t')
'''
loaders = DataLoader(
            train_fn='./data/sms.maxlen.uniq.shuf.train.tsv',
            batch_size = batch_size,
            valid_ratio = .2,
            device = 0,
            max_vocab = 999999,
            min_freq = 5
)

test_loaders = DataLoader(
            train_fn='./data/sms.maxlen.uniq.shuf.test.tsv',
            batch_size = batch_size,
            valid_ratio = .01,
            device = 0,
            max_vocab = 999999,
            min_freq = 5
)
vocab_size = len(loaders.text.vocab)
num_classes = len(loaders.label.vocab)

model = RNN(vocab_size, word_vec_size, hidden_size, num_classes, num_layers, dropout_p).to(device)

loss_func = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

total_step = len(loaders.train_loader)
for epoch in range(num_epochs):
    for i, data in enumerate(loaders.train_loader):
        texts = data.text.to(device)
        labels = data.label.to(device)
        #print("[%d]"%i)
        outputs = model(texts)
        loss = loss_func(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 10 == 0:
            print("Epoch [{}/{}], step [{}/{}], Loss : {:.4f}, Accr: {:.2f}".format(epoch+1,num_epochs, i+1, total_step, loss.item(), ComputeAccr(loaders.valid_loader, model)))
        







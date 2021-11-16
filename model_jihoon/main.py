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
num_epochs = 100

word_vec_size = 256
dropout_p = 0.3

hidden_size = 512
num_layers = 4

learning_rate = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PREPROCESS = True

if PREPROCESS:
    # read original tsv
    df = pd.read_csv('./data/draw.tsv', sep='\t')

    # preprocess - description 
    _description = df['description']
    res = []
    for i in range(_description.shape[0]):
        try:
            arr = _description[i].split('.')
            if len(arr) > 5:
                if len(arr[-1]) > 2:
                    res.append(arr)
                else:
                    res.append(arr[:-1])
        except:
            break


    # for labeling
    _df = df.copy()
    _df['h_length'] = _df['h_length'].apply(lambda x:"hl"+str(int(x))) # description X
    _df['h_bang'] = _df['h_bang'].apply(lambda x:"hb"+str(int(x)))
    _df['h_curl'] = _df['h_curl'].apply(lambda x:"hc"+str(int(x)))
    _df['e_shape'] = _df['e_shape'].apply(lambda x:"es"+str(int(x))) # description X
    _df['f_shape'] = _df['f_shape'].apply(lambda x:"fs"+str(int(x))) 
    _df['sex'] = _df['sex'].apply(lambda x:"s"+str(int(x)))
    _df['nose'] = _df['nose'].apply(lambda x:"n"+str(int(x))) # description X

    # arrange data for train
    train_df = pd.DataFrame(columns=['label', 'description'])
    for i in range(len(res)):
        idx = pd.DataFrame(res[i])[0].str.contains('bang').argmax()
        train_df = train_df.append({'label' : _df['h_bang'][i], 'description' : str(res[i][idx])}, ignore_index=True)
        
        idx = pd.DataFrame(res[i])[0].str.contains('curl|straight').argmax()
        train_df = train_df.append({'label' : _df['h_curl'][i], 'description' : str(res[i][idx])}, ignore_index=True)
        
        idx = pd.DataFrame(res[i])[0].str.contains('face').argmax()
        train_df = train_df.append({'label' : _df['f_shape'][i], 'description' : str(res[i][idx])}, ignore_index=True)
        
        idx = pd.DataFrame(res[i])[0].str.contains('woman|man|girl|boy').argmax()
        train_df = train_df.append({'label' : _df['sex'][i], 'description' : str(res[i][idx])}, ignore_index=True)
    
    # shuffle - prob not necessary
    df_shuffled = train_df.sample(frac=1).reset_index(drop=True)

    # for overfitting & small dataset
    # - train dataset
    s, e = 0, int(df_shuffled.shape[0] * 1)
    df_train = pd.DataFrame({'label' : df_shuffled['label'][s:e],
                            'description' : df_shuffled['description'][s:e]})

    # - test dataset
    e = int(df_shuffled.shape[0] * 0.9)
    s, e = e, e + int(df_shuffled.shape[0] * 0.1)
    df_test = pd.DataFrame({'label' : df_shuffled['label'][s:e],
                            'description' : df_shuffled['description'][s:e]})

    # save data
    df_train.to_csv('./data/draw.train.tsv', header=False, index=False, sep='\t')
    df_test.to_csv('./data/draw.test.tsv', header=False, index=False, sep='\t')

# load data
loaders = DataLoader(
        train_fn='./data/draw.train.tsv',
        batch_size = batch_size,
        valid_ratio = .1,
        device = 0,
        max_vocab = 999999,
        min_freq = 5
)

test_loaders = DataLoader(
            train_fn='./data/draw.test.tsv',
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


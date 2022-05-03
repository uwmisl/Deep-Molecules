#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd

import os
import re
import random


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 600
DATA_DIR = './data/PDB-2021AUG02.csv'

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# In[3]:


def load_dataset(max_length, data_dir=''):
    print ("loading dataset...")
    data = pd.read_csv(data_dir)
    dirname = data_dir.split('.')
    dirfilename = (dirname[1].split('/'))[-1]
    print(dirfilename)
    lines = list(set(data['SEQUENCE'].tolist()))
    lines = [l for l in lines if ('X' not in l)]
    
    
    
    lines = [l for l in lines if (len(l) <= max_length)]
    lines = [tuple(l + '0'*(MAX_LENGTH - len(l))) for l in lines] # pad with 0
    print("loaded {} lines in dataset".format(len(lines)))
    np.random.shuffle(lines) 
    return lines


# In[4]:


class Lang:
    def __init__(self, name):
        self.name = name
        self.char2index = {}
        self.char2count = {}
        self.char2word = {}
        self.n_chars = 0

    def addSequence(self, seq):
        for c in list(seq):
            self.addChar(c)

    def addChar(self, c):
        if c not in self.char2index:
            self.char2index[c] = self.n_chars
            self.char2count[c] = 1
            self.char2word[self.n_chars] = c
            self.n_chars += 1
        else:
            self.char2count[c] += 1


# In[5]:


def prepare_data(max_len=MAX_LENGTH, data_dir=DATA_DIR):
    lines = load_dataset(max_len, data_dir)
    lang = Lang("PDB")
    for line in lines:
        lang.addSequence(line)
    retlines = []
    for s in lines:
        retlines.append(F.one_hot(tensorFromSequence(lang, s), num_classes=lang.n_chars).float())
    return (lang, lang, [[s, s] for s in retlines])

def indexesFromSequence(lang, sequence):
    return [lang.char2index[c] for c in list(sequence)]

def tensorFromSequence(lang, sequence):
    indexes = indexesFromSequence(lang, sequence)
    return torch.tensor(indexes, dtype=torch.long, device=device)

input_lang, output_lang, pairs = prepare_data()

def tensorsFromPair(pair):
    input_tensor = tensorFromSequence(input_lang, pair[0])
    target_tensor = tensorFromSequence(output_lang, pair[1])
    return (input_tensor, target_tensor)


# In[6]:


# class EncoderRNN(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers=1):
#         super(EncoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.embedding = nn.Embedding(input_size, hidden_size)
#         self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers)

#     def forward(self, input, hidden):
#         embedded = self.embedding(input).view(1, 1, -1)
#         output = embedded
#         if hidden.shape[0] != self.num_layers:
#             hidden = hidden.repeat(self.num_layers, 1, 1)
#         output, hidden = self.gru(output, hidden)
#         return output, hidden

#     def initHidden(self):
#         return torch.zeros(1, 1, self.hidden_size, device=device)

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.layers = nn.Sequential(nn.Linear(input_size, hidden_size), 
                                    nn.ReLU(), 
                                    nn.Linear(hidden_size, hidden_size), 
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, hidden_size), 
                                    nn.ReLU())

    def forward(self, input):
        output = self.layers(input)
        return output

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# In[7]:


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.layers = nn.Sequential(nn.Linear(hidden_size, hidden_size), 
                                    nn.ReLU(), 
                                    nn.Linear(hidden_size, hidden_size), 
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, hidden_size), 
                                    nn.ReLU())
        self.out = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.layers(input)
        output = self.sigmoid(self.out(output))
        return output

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# In[8]:


# class Model(nn.Module):
#     def __init__(self, input_size, encoding_size, hidden=[], h_act=nn.ReLU(), out_act=nn.Tanh()):
#         super(Model, self).__init__()
#         self.encoder = Encoder(input_size, encoding_size, hidden, h_act, out_act)
#         self.decoder = Decoder(encoding_size, input_size, hidden, h_act)
    
#     def forward(self, x):
#         seq_len = x.shape[0]
#         x = self.encoder(x)
#         x = self.decoder(x, seq_len)
#         return x


# In[32]:


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    encoder_hidden = encoder(input_tensor)
    decoder_output = decoder(encoder_hidden)
    loss = criterion(decoder_output, target_tensor)
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return decoder_output, loss.item()


# In[43]:


def trainIters(encoder, decoder, n_iters, epochs, print_every=1, plot_every=1, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    for e in range(1, epochs + 1):
        training_pairs = [random.choice(pairs) for i in range(n_iters)]
        for iter in range(1, n_iters + 1):
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0].view(-1, MAX_LENGTH * input_lang.n_chars)
            target_tensor = training_pair[1].view(-1, MAX_LENGTH * input_lang.n_chars)

            output_tensor, loss = train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss
            
            with torch.no_grad():
                if iter % 1000 == 0:
                    print('\tIteration %d Loss: %.4f' % (iter, loss))
                    print('\t\t Output:', torch.argmax(output_tensor.view(MAX_LENGTH, input_lang.n_chars), dim=1))
                    print('\t\t Input:', torch.argmax(training_pair[1].view(MAX_LENGTH, input_lang.n_chars), dim=1))

        if e % print_every == 0:
            print_loss_avg = print_loss_total / n_iters
            print_loss_total = 0
            print('%s (%d %d%%) Loss: %.4f' % (timeSince(start, e / epochs),
                                         e, e / epochs * 100, print_loss_avg))

        if e % plot_every == 0:
            plot_loss_avg = plot_loss_total / n_iters
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0


# In[ ]:


hidden_size = 500
encoder1 = EncoderRNN(input_lang.n_chars * MAX_LENGTH, hidden_size, num_layers=1).to(device)
decoder1 = DecoderRNN(hidden_size, output_lang.n_chars * MAX_LENGTH, num_layers=1).to(device)

trainIters(encoder1, decoder1, 100000, 500)


# In[ ]:





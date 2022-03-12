#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable

import pandas as pd
from itertools import groupby
import collections
from sklearn.preprocessing import OneHotEncoder
import os, math, glob, argparse
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np


# In[2]:


cuda = True if torch.cuda.is_available() else False


# ## Utitilites

# In[3]:


def _sample_gumbel(shape, eps=1e-20, out=None):
    """
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
    return - torch.log(eps - torch.log(U + eps))


def _gumbel_softmax_sample(logits, temp=1, eps=1e-20):
    """
    Draw a sample from the Gumbel-Softmax distribution
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    dims = logits.dim()
    gumbel_noise = _sample_gumbel(logits.size(), eps=eps, out=logits.data.new())
    y = logits + Variable(gumbel_noise)
    return F.softmax(y / temp, dims - 1)


def gumbel_softmax(logits, temp=1, hard=False, eps=1e-20):
    """
    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temp: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes
    Constraints: - this implementation only works on batch_size x num_features tensor for now
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    shape = logits.size()
    assert len(shape) == 2
    y_soft = _gumbel_softmax_sample(logits, temp=temp, eps=eps)
    if not hard:
        return y_soft
    _, k = y_soft.data.max(-1)
    y_hard = logits.data.new(*shape).zero_().scatter_(-1, k.view(-1, 1), 1.0)
    return Variable(y_hard - y_soft.data) + y_soft

def load_dataset(max_length, max_n_examples, max_vocab_size=2048, data_dir=''):
    print ("loading dataset...")
    data = pd.read_csv(data_dir)
    dirname = data_dir.split('.')
    dirfilename = (dirname[1].split('/'))[-1]
    print(dirfilename)
    lines = list(set(data['SEQUENCE'].tolist()))
    make_graphs(lines, dirfilename)
    lines = [l for l in lines if ('X' not in l)]
    make_graphs(lines, dirfilename + '_no_X')
    
    
    
    lines = [l for l in lines if (len(l) <= max_length)]
    lines = [tuple(l + '0'*(max_length - len(l))) for l in lines] # pad with 0
    print("loaded {} lines in dataset".format(len(lines)))    
    
    np.random.shuffle(lines)
    counts = collections.Counter(char for line in lines for char in line)

    charmap = {}
    inv_charmap = []

    for char,count in counts.most_common(max_vocab_size-1):
        if char not in charmap:
            charmap[char] = len(inv_charmap)
            inv_charmap.append(char)
            
    return lines, charmap, inv_charmap

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def decode_one_seq(img, letter_dict):
    seq = ''
    for row in range(len(img)):
        on = np.argmax(img[row,:])
        seq += letter_dict[on]
    return seq

def make_graphs(lines, name):
    lines = [s.replace('0', '') for s in lines]
    # Make graph -- length
    lengths = [len(s) for s in lines]
    count_lengths = collections.Counter(lengths)
    fig, ax = plt.subplots(3, figsize=(15,30))
    ax[0].bar(count_lengths.keys(), count_lengths.values())
    ax[0].grid()
    ax[0].set_title('Frequency of Sequence Lengths')
    ax[0].set_xlabel('Length')
    ax[0].set_ylabel('Frequency')
    ax[1].bar(count_lengths.keys(), count_lengths.values())
    ax[1].grid()
    ax[1].set_title('Frequency of Sequence Lengths')
    ax[1].set_xlabel('Length')
    ax[1].set_ylabel('Frequency')
    ax[1].set_xlim((0, 1000))
    ax[2].bar(count_lengths.keys(), count_lengths.values())
    ax[2].grid()
    ax[2].set_title('Frequency of Sequence Lengths')
    ax[2].set_xlabel('Length')
    ax[2].set_ylabel('Frequency')
    ax[2].set_xlim((0, 600))
    fig.savefig('./graphs/{0}_Seq_Lengths.png'.format(name))
    # Make graph -- freq
    freq = {}
    for s in lines:
      for k, v in collections.Counter(s).items():
          if k not in freq:
            freq[k] = 0
          freq[k] += v
    fig, ax = plt.subplots(1, figsize=(15,10))
    ofreq = collections.OrderedDict(sorted(freq.items()))
    ax.bar(ofreq.keys(), ofreq.values())
    ax.grid()
    ax.set_title('Frequency of Amino Acids')
    ax.set_xlabel('Amino Acids')
    ax.set_ylabel('Frequency')
    fig.savefig('./graphs/{0}_Seq_Freq.png'.format(name))
    # Make graph -- dups
    dups = {}
    for s in lines:
      for k, v in collections.Counter(zip(s, s[1:])).items():
        if k[0] == k[1]:
          if k[0] not in dups:
            dups[k[0]] = 0
          dups[k[0]] += v
    fig, ax = plt.subplots(1, figsize=(15,10))
    odups = collections.OrderedDict(sorted(dups.items()))
    ax.bar(odups.keys(), odups.values())
    ax.grid()
    ax.set_title('Frequency of Duplicate Amino Acids')
    ax.set_xlabel('Amino Acids')
    ax.set_ylabel('Frequency')
    fig.savefig('./graphs/{0}_Seq_Dups.png'.format(name))
    # Make graph -- runs
    runs = np.zeros(2000)
    for s in lines:
        ch = s[0]
        count = 1
        for i in range(1, len(s)):
            if s[i] == ch:
                count += 1
            else:
                runs[count] += 1
                ch = s[i]
                count = 1
    runs = runs / len(lines)
            
    fig, ax = plt.subplots(1, figsize=(15,10))
    odups = collections.OrderedDict(sorted(dups.items()))
    ax.bar(np.arange(0, 2000), runs)
    ax.grid()
    ax.set_title('Frequency of Runs Lengths')
    ax.set_xlabel('Run Length')
    ax.set_ylabel('log(Frequency)')
    ax.set_yscale("log")
    fig.savefig('./graphs/{0}_Seq_Runs.png'.format(name))
    plt.close('all')


# ## FBGAN

# In[4]:


class Generator(nn.Module):
    def __init__(self, seq_len, batch_size, dim, num_classes):
        super(Generator, self).__init__()

        def block(dim):
            layers = []
            layers.append(nn.Conv1d(dim, dim, 5, padding=2))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv1d(dim, dim, 5, padding=2))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv1d(dim, dim, 5, padding=2))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv1d(dim, dim, 5, padding=2))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv1d(dim, dim, 5, padding=2))
            layers.append(nn.ReLU(inplace=True))
            return layers
        
        self.L = nn.Linear(128, dim*seq_len)
        self.C = nn.Conv1d(dim, num_classes, 1)
        self.model = nn.Sequential(
            *block(dim),
            *block(dim),
            *block(dim),
            *block(dim),
            *block(dim),
            *block(dim),
            *block(dim),
            *block(dim),
            *block(dim),
            *block(dim)
        )
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.dim = dim;
        self.num_classes = num_classes;

    def forward(self, z):
        out = self.L(z) # (batch_size, dim * seq_len)
        out = out.view(-1, self.dim, self.seq_len) # (batch_size, dim, seq_len)
        out = self.model(out)
        out = self.C(out) # (batch_size, num_classes, seq_len)
        out = out.transpose(1, 2)
        size = out.size()
        out = out.contiguous()
        out = out.view(self.batch_size*self.seq_len, -1)
        out = gumbel_softmax(out, 0.5)
        return out.view(size)


# In[5]:


class Discriminator(nn.Module):
    def __init__(self, seq_len, batch_size, dim, num_classes):
        super(Discriminator, self).__init__()
        
        def block(dim):
            layers = []
            layers.append(nn.Conv1d(dim, dim, 5, padding=2))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv1d(dim, dim, 5, padding=2))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv1d(dim, dim, 5, padding=2))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv1d(dim, dim, 5, padding=2))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv1d(dim, dim, 5, padding=2))
            layers.append(nn.ReLU(inplace=True))
            return layers
        
        self.C = nn.Conv1d(num_classes, dim, 1)
        
        self.model = nn.Sequential(
            *block(dim),
            *block(dim),
            *block(dim),
            *block(dim)
        )
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.dim = dim;
        self.num_classes = num_classes;
        self.linear = nn.Linear(seq_len*dim, 1)
        self.activation = nn.Sigmoid()

    def forward(self, input):
        out = input.transpose(1, 2)
        out = self.C(out)
        out = self.model(out)
        out = out.view(-1, self.seq_len*self.dim)
        out = self.linear(out)
        out = self.activation(out)
        return out


# ## Train Model

# In[8]:


class FBGAN():
    def __init__(self, batch_size=64, lr=0.00001, num_epochs=100, seq_len = 400, data_dir='./data/PDB-2021AUG02.csv',         run_name='test02', hidden=256, d_steps = 10):
        self.hidden = hidden
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = num_epochs
        self.seq_len = seq_len
        self.d_steps = d_steps
        self.g_steps = 1
        self.lamda = 10
        self.sample_dir = './samples/' + run_name + "/"
        self.load_data(data_dir)
        if not os.path.exists(self.sample_dir): os.makedirs(self.sample_dir)
        self.use_cuda = True if torch.cuda.is_available() else False
        self.build_model()

    def build_model(self):
        self.G = Generator(self.seq_len, self.batch_size, self.hidden, len(self.charmap))
        self.D = Discriminator(self.seq_len, self.batch_size, self.hidden, len(self.charmap))
        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()
        print(self.G)
        print(self.D)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.9))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.5, 0.9))

    def load_data(self, datadir):
        max_examples = 1e6
        lines, self.charmap, self.inv_charmap = load_dataset(
            max_length=self.seq_len,
            max_n_examples=max_examples,
            data_dir=datadir
        )
        self.data = lines

    def calc_gradient_penalty(self, real_data, fake_data):
        alpha = torch.rand(self.batch_size, 1, 1)
        alpha = alpha.view(-1,1,1)
        alpha = alpha.expand_as(real_data)
        alpha = alpha.cuda() if self.use_cuda else alpha
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.cuda() if self.use_cuda else interpolates
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = self.D(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda() \
                                  if self.use_cuda else torch.ones(disc_interpolates.size()),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1).norm(2,dim=1) - 1) ** 2).mean() * self.lamda
        return gradient_penalty
    
    def train_model(self):
        init_epoch = 1
        total_iterations = 4000
        d_fake_losses, d_real_losses, grad_penalties = [],[],[]
        G_losses, D_losses, W_dist = [],[],[]

        one = one = torch.tensor(1, dtype=torch.float)
        one = one.cuda() if self.use_cuda else one
        one_neg = one * -1

        table = np.arange(len(self.charmap)).reshape(-1, 1)
        one_hot = OneHotEncoder()
        one_hot.fit(table)

        i = 0
        for epoch in range(self.n_epochs):
            n_batches = int(len(self.data)/self.batch_size)
            for idx in range(n_batches):
                _data = np.array(
                    [[self.charmap[c] for c in l] for l in self.data[idx*self.batch_size:(idx+1)*self.batch_size]],
                    dtype='int32'
                )
                data_one_hot = one_hot.transform(_data.reshape(-1, 1)).toarray().reshape(self.batch_size, -1, len(self.charmap))
                real_data = torch.Tensor(data_one_hot)
                real_data = to_var(real_data)
                for p in self.D.parameters():  # reset requires_grad
                    p.requires_grad = True  # they are set to False below in netG update
                d_real_acc = 0
                d_fake_acc = 0
                for _ in range(self.d_steps):
                    self.D.zero_grad()
                    d_real_pred = self.D(real_data)
                    d_real_err = torch.mean(d_real_pred) #want to push d_real as high as possible
                    d_real_acc += d_real_err.item()
                    d_real_err.backward(one_neg)

                    z_input = to_var(torch.randn(self.batch_size, 128))
                    d_fake_data = self.G(z_input).detach()
                    d_fake_pred = self.D(d_fake_data)
                    d_fake_err = torch.mean(d_fake_pred) #want to push d_fake as low as possible
                    d_fake_acc += d_fake_err.item()
                    d_fake_err.backward(one)

                    gradient_penalty = self.calc_gradient_penalty(real_data.data, d_fake_data.data)
                    gradient_penalty.backward()

                    d_err = d_fake_err - d_real_err + gradient_penalty
                    self.D_optimizer.step()
                d_real_acc /= self.d_steps
                d_fake_acc = (d_fake_acc / self.d_steps)
                print("Accuracy on real data:", 1 - d_real_acc)
                print("Accuracy on fake data:", 1 - d_fake_acc)
                # Append things for logging
                d_fake_np, d_real_np, gp_np = (d_fake_err.data).cpu().numpy(),                         (d_real_err.data).cpu().numpy(), (gradient_penalty.data).cpu().numpy()
                grad_penalties.append(gp_np)
                d_real_losses.append(d_real_np)
                d_fake_losses.append(d_fake_np)
                D_losses.append(d_fake_np - d_real_np + gp_np)
                W_dist.append(d_real_np - d_fake_np)
                # Train G
                for p in self.D.parameters():
                    p.requires_grad = False  # to avoid computation

                self.G.zero_grad()
                z_input = to_var(torch.randn(self.batch_size, 128))
                g_fake_data = self.G(z_input)
                dg_fake_pred = self.D(g_fake_data)
                g_err = -torch.mean(dg_fake_pred)
                g_err.backward()
                self.G_optimizer.step()
                G_losses.append((g_err.data).cpu().numpy())
                if i % 100 == 99:
                    self.sample(i)
                    summary_str = 'Iteration [{}/{}] - loss_d: {}, loss_g: {}, w_dist: {}, grad_penalty: {}'                        .format(i, total_iterations, (d_err.data).cpu().numpy(),
                        (g_err.data).cpu().numpy(), ((d_real_err - d_fake_err).data).cpu().numpy(), gp_np)
                    print(summary_str)
                if i % 300 == 99:
                    self.sample(i, save_to_file=False)
                i += 1
                print("Iteration:" + str(i))
            np.random.shuffle(self.data)

    def sample(self, epoch, save_to_file=True):
        if save_to_file:
            z = to_var(torch.randn(self.batch_size, 128))
            self.G.eval()
            torch_seqs = self.G(z)
            seqs = (torch_seqs.data).cpu().numpy()
            decoded_seqs = [decode_one_seq(seq, self.inv_charmap)+"\n" for seq in seqs]
            with open(self.sample_dir + "sampled_{0:05}.txt".format(epoch), 'w+') as f:
                f.writelines(decoded_seqs)
        else:
            z = to_var(torch.randn(64, 128))
            self.G.eval()
            torch_seqs = self.G(z)
            seqs = (torch_seqs.data).cpu().numpy()
            decoded_seqs = [decode_one_seq(seq, self.inv_charmap) for seq in seqs]
            make_graphs(decoded_seqs, "sampled_{0:05}".format(epoch))
        self.G.train()


# In[ ]:


model = FBGAN()
model.train_model()


# In[ ]:





# In[ ]:





# In[ ]:





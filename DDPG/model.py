
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self, nb_pos,nb_laser, nb_actions, hidden1=64, hidden2=64, init_w=3e-3, layer_norm = True):
        super(Actor, self).__init__()
        self.layer_norm = layer_norm
        self.fc1 = nn.Linear(nb_laser, hidden1)
        self.fc2 = nn.Linear(hidden1+nb_pos, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions, bias = False)
        self.relu = nn.ReLU()
        self.softsign = nn.Softsign()
        if self.layer_norm :
            self.LN1 = nn.LayerNorm(hidden1)
            self.LN2 = nn.LayerNorm(hidden2)
        
        
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, x):
        pos,laser = x
        out = self.fc1(laser)
        if self.layer_norm :
            out = self.LN1(out)
        out = self.relu(out)
        out = self.fc2(torch.cat([out,pos],1))
        if self.layer_norm :
            out = self.LN2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.softsign(out)
        return out

class Critic(nn.Module):
    def __init__(self, nb_pos,nb_laser, nb_actions, hidden1=64, hidden2=64, init_w=3e-3, layer_norm = True):
        super(Critic, self).__init__()
        self.layer_norm = layer_norm
        self.fc1 = nn.Linear(nb_laser, hidden1)
        self.fc2 = nn.Linear(hidden1+nb_pos+nb_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        if self.layer_norm :
            self.LN1 = nn.LayerNorm(hidden1)
            self.LN2 = nn.LayerNorm(hidden2)
        
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, xs):
        pos,laser, a = xs
        out = self.fc1(laser)
        if self.layer_norm :
            out = self.LN1(out)
        out = self.relu(out)
        out = self.fc2(torch.cat([out,pos,a],1))
        if self.layer_norm :
            out = self.LN2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
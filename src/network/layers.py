import torch
import numpy as np
from torch.nn import ModuleList, Module, Sequential, Linear, Dropout, ReLU, CosineSimilarity, Softmax

class SelfAttention(Module):

    def __init__(self, d):
        super(SelfAttention, self).__init__()
        self.d = d
        self.W_q = Linear(2*d, 2*d)
        self.W_k = Linear(2*d, 2*d)
        self.W_v = Linear(2*d, 2*d)
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        pre_alpha = torch.matmul(Q,K.permute(0,2,1))/np.sqrt(2*self.d)
        alpha = self.softmax(pre_alpha)
        h_hat = torch.matmul(alpha,V)
        return h_hat

class NeighborEncoder(Module):

    def __init__(self, d, attention_num):
        super(NeighborEncoder, self).__init__()
        self.attention_layers = ModuleList([SelfAttention(d) for _ in range(attention_num)])
        self.fc = Sequential(Linear(2*d, 4*2*d),
                            ReLU(),
                            Linear(4*2*d, 2*d),
                            ReLU())

    def forward(self, H):
        x = H
        for layer in self.attention_layers:
            x = layer(x)
        x = self.fc(x)
        return x

class Similarity(Module):

    def __init__(self):
        super(Similarity,self).__init__()
        self.cs = CosineSimilarity(dim=-1, eps=1e-8)

    def forward(self, a, b):
        return (1+self.cs(a,b))/2
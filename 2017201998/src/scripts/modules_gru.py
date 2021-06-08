import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self, n_embs, n_hidden, vocab_size, dropout, g_max_len):
        super(Model, self).__init__()
        
        # initlize parameters
        self.n_embs = n_embs
        self.n_hidden = n_hidden
        self.vocab_size = vocab_size
        self.g_max_len = g_max_len
        self.dropout = dropout

        self.word_embedding = nn.Embedding(vocab_size, n_embs)
        self.text_encoder = GRU_Encoder(n_embs=self.n_embs, dim_ff=self.n_hidden,dropout=dropout)
        self.decoder = GRU_Decoder(self.n_embs, self.n_embs, self.n_hidden, self.dropout)
        self.output_layer = nn.Linear(self.n_hidden, self.vocab_size)
        self.criterion = nn.CrossEntropyLoss(reduce=False, size_average=False, ignore_index=0)


    def encode_text(self, Text):
        text_embs = self.word_embedding(Text)
        L, G = self.text_encoder(text_embs) # use LSTM last hidden state as global feature
        return G, L[:,:-1,:]

    def decode(self, Title, G, L):
        text_embs = self.word_embedding(Title)
        out = self.decoder(text_embs, G, L)
        out = self.output_layer(out)
        return out

    def forward(self, Abstract, Title):
        T, Text = self.encode_text(Abstract)
        outs = self.decode(Title[:,:-1], T, Text)
        
        Title = Title.t()
        outs= outs.transpose(0, 1)
        loss = self.criterion(outs.contiguous().view(-1, self.vocab_size), Title[1:].contiguous().view(-1))
        return torch.mean(loss)

    def generate(self, Abstract, beam_size=1):
        T, Text = self.encode_text(Abstract.unsqueeze(0))
        Des = Variable(torch.ones(1, 1).long()).cuda()
        with torch.no_grad():
            for i in range(self.g_max_len):
                # embs = self.word_embedding(Des)
                out = self.decode(Des, T, Text)
                prob = out[:, -1]
                _, next_w = torch.max(prob, dim=-1, keepdim=True)
                next_w = next_w.data
                Des = torch.cat([Des, next_w], dim=-1)
        return Des[:, 1:]


class WordEmbeddings(nn.Module):
    def __init__(self, n_emb, vocabs):
        super(WordEmbeddings, self).__init__()
        self.n_emb = n_emb
        self.word2vec = nn.Embedding(vocabs, n_emb)

    def forward(self, x):
        return self.word2vec(x)

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class GRU_Encoder(nn.Module):
    def __init__(self, n_embs, dim_ff, dropout):
        super(GRU_Encoder, self).__init__()
        self.rnn = nn.GRU(
            input_size=n_embs,
            hidden_size=dim_ff,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )

    def forward(self, x):
        # print('Encoder ', x.size())
        output, hidden = self.rnn(x,None)
        # print(output.size(),hidden.size())
        return output, hidden

class GRU_Decoder(nn.Module):
    def __init__(self, l_dims, n_embs, hidden_size, dropout):
        super(GRU_Decoder, self).__init__()
        self.input_size = n_embs
        self.hidden_size = hidden_size
        self.l_dims = l_dims
        self.dropout = dropout

        self.gru_l0 = nn.GRUCell(self.input_size, self.hidden_size, bias=False)
        self.gru_l1 = nn.GRUCell(self.hidden_size, self.hidden_size, bias=False)
        self.gru_l2 = nn.GRUCell(self.input_size, self.hidden_size, bias=False)
        self.gru_l3 = nn.GRUCell(self.hidden_size, self.hidden_size, bias=False)
        self.linear = nn.Linear(2*self.hidden_size, self.hidden_size, bias=False)
        self.norm = LayerNorm(self.hidden_size)
        self.drop = nn.Dropout(self.dropout)

    def attn(self, l, h):
        """
        args:
            l: additional local information [B, N, l_dims = hidden_size]
            h: the current time step hidden state [B, hidden_size]

        return:
            h_: this time step context [B, hidden_size]
        """
        h = h.unsqueeze(1)
        weights = torch.bmm(h, l.transpose(1,2))
        attn_weights = F.softmax(weights, dim=-1)
        h_ = torch.bmm(attn_weights, l)
        return h.squeeze(1)

    def step_decoding(self, x, L, h):
        #print(x.size(),h.size())
        
        h1 = self.gru_l0(x, h[0])
        h_ = self.drop(h1)
        h_= self.attn(L, h_)
        h1 = self.linear(torch.cat([h[0], h_], dim=1)).tanh()
        h = h1.unsqueeze(0) # [layers_num = 8, B, hidden_size]
        '''
        h1 = self.gru_l0(x, h[0])
        h2 = self.gru_l1(h1, h[1])
        h_ = self.drop(h2)
        h_= self.attn(L, h_)
        h1 = self.linear(torch.cat([h[0], h_], dim=1)).tanh()
        h = torch.cat([h1.unsqueeze(0),h2.unsqueeze(0)],dim=0,out=None) # [layers_num = 2, B, hidden_size]
        
        h1 = self.gru_l0(x, h[0])
        h2 = self.gru_l1(h1, h[1])
        h3 = self.gru_l2(h2, h[2])
        h_ = self.drop(h3)
        h_= self.attn(L, h_)
        h1 = self.linear(torch.cat([h[0], h_], dim=1)).tanh()
        h = torch.cat([h1.unsqueeze(0),h2.unsqueeze(0),h3.unsqueeze(0)],dim=0,out=None)
        
        h1 = self.gru_l0(x, h[0])
        h2 = self.gru_l1(h1, h[1])
        h3 = self.gru_l2(h2, h[2])
        h4 = self.gru_l3(h3, h[3])
        h_ = self.drop(h4)
        h_= self.attn(L, h_)
        h1 = self.linear(torch.cat([h[0], h_], dim=1)).tanh()
        h = torch.cat([h1.unsqueeze(0),h2.unsqueeze(0),h3.unsqueeze(0),h4.unsqueeze(0)],dim=0,out=None)
        '''
        #h1 = self.drop(h1)
        return h

    def forward(self, X, G, L):
        output = []
        h = G
        for t in range(X.size(1)):
            # one step decoding
            x = X[:, t, :]
            h = self.step_decoding(x, L, h)
            output.append(h[0])
        output = torch.stack(output, dim=1)  # [B, MAX_LEN, hidden_size]
        
        #print(output)
        return self.norm(output)




import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self, n_embs, n_hidden, vocab_size, dropout, g_max_len, input_max_len):
        super(Model, self).__init__()
        
        # initlize parameters
        self.n_embs = n_embs
        self.n_hidden = n_hidden
        self.vocab_size = vocab_size
        self.g_max_len = g_max_len
        self.dropout = dropout

        self.word_embedding = nn.Embedding(vocab_size, n_embs)
        self.text_encoder = Transformer(n_embs=n_embs, dim_ff=n_hidden, n_head=8, n_block=4, dropout=dropout)
        self.text_project = nn.Sequential(nn.Linear(n_embs, n_hidden), nn.Sigmoid())
        self.position_embedding = PositionalEncoder(n_hidden, input_max_len, dropout)
        self.decoder = GRU_Decoder(self.n_embs, self.n_embs, self.n_hidden, self.dropout)
        self.output_layer = nn.Linear(self.n_hidden, self.vocab_size)
        self.criterion = nn.CrossEntropyLoss(reduce=False, size_average=False, ignore_index=0)


    def encode_text(self, Text):
        text_embs = self.word_embedding(Text)
        text_embs = self.text_project(text_embs)
        #text_embs = self.position_embedding(text_embs)
        L= self.text_encoder(text_embs)
        G = torch.mean(L, dim=1, keepdim=True)
        G = G.transpose(0,1)
        return G, L

    def decode(self, Title, G, L):
        text_embs = self.word_embedding(Title)
        out = self.decoder(text_embs, G, L)
        out = self.output_layer(out)
        return out

    def forward(self, Abstract, Title):
        T, Text = self.encode_text(Abstract)
        T = T.repeat(2,1,1)
        outs = self.decode(Title[:,:-1], T, Text)
        
        Title = Title.t()
        outs= outs.transpose(0, 1)
        loss = self.criterion(outs.contiguous().view(-1, self.vocab_size), Title[1:].contiguous().view(-1))
        return torch.mean(loss)

    def generate(self, Abstract, beam_size=1):
        T, Text = self.encode_text(Abstract.unsqueeze(0))
        T = T.repeat(2,1,1)
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

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len, dropout):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
  
    def forward(self, x, step=None):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        if seq_len == 1 and step is not None:
            pe = Variable(self.pe[:,step-1], requires_grad=False).cuda()
        else:
            pe = Variable(self.pe[:,:seq_len], requires_grad=False).cuda()
        x = x + pe
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, n_embs, dim_ff, n_head, dropout, n_block):
        super(Transformer, self).__init__()
        self.norm = LayerNorm(n_embs)
        self.layers = nn.ModuleList([AttnBlock(n_embs, dim_ff, n_head, dropout) for _ in range(n_block)])


    def forward(self, x):
        # print('Encoder ', x.size())
        x = self.norm(x)
        for layer in self.layers:
            x = layer(x, x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, dim, dim_ff, n_head, dropout):
        super(AttnBlock, self).__init__()
        self.attn = MultiHeadAttention(dim, n_head, dropout)
        self.feed_forward = PositionwiseFeedForward(dim, dim_ff, dropout)
        self.sublayer = nn.ModuleList([SublayerConnection(dim, dropout) for _ in range(2)])

    def forward(self, x, m):
        x = self.sublayer[0](x, lambda x: self.attn(x, m, m))
        return self.sublayer[1](x, self.feed_forward)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_head, dropout):
        super(MultiHeadAttention, self).__init__()
        assert dim % n_head == 0
        self.d_k = dim // n_head
        self.n_head = n_head
        self.linears = nn.ModuleList([nn.Linear(dim, dim) for _ in range(4)]) 
        #其中0、1、2是Q、K、V输入的线性变换，3是最后连接输出的线性变换
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        weights = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_k)

        if mask is not None:
            weights = weights.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(weights, dim=-1)

        if dropout is not None:
            p_attn = self.dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [l(x).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.n_head * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, dim, dim_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(dim, dim_ff)
        self.w_2 = nn.Linear(dim_ff, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

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
        h2 = self.gru_l1(h1, h[1])
        #h3 = self.gru_l2(h2, h[2])
        #h4 = self.gru_l3(h3, h[3])
        h_ = self.drop(h2)
        h_= self.attn(L, h_)
        h1 = self.linear(torch.cat([h[0], h_], dim=1)).tanh()
        #h = h1.unsqueeze(0) # [layers_num = 8, B, hidden_size]
        h = torch.cat([h1.unsqueeze(0),h2.unsqueeze(0)],dim=0,out=None) # [layers_num = 2, B, hidden_size]
        #h = torch.cat([h1.unsqueeze(0),h2.unsqueeze(0),h3.unsqueeze(0)],dim=0,out=None)
        #h = torch.cat([h1.unsqueeze(0),h2.unsqueeze(0),h3.unsqueeze(0),h4.unsqueeze(0)],dim=0,out=None)
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




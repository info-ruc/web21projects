
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

from time import time
import pickle
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

vocabulary = pickle.load(open('/home/yutong_bai/doc_profile/DocPro/data/vocab.dict', 'rb'))

parser = ArgumentParser("deep_personalization",
                        formatter_class=ArgumentDefaultsHelpFormatter,
                        conflict_handler='resolve')

parser.add_argument('--num_epoch', default=20, type=int,
                    help='The number of the epochs.')
args = parser.parse_args()

cudaid = 1
CUDA = True



class GRU(torch.nn.Module):

    def __init__(self, input_size, hidden_size, id_dict_size=7, withuser=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.proj_q = nn.Linear(id_dict_size, self.hidden_size)
        self.withuser = withuser
        self.gru_cell = nn.GRUCell(input_size=input_size, hidden_size=self.hidden_size) # 规定输入维度, 隐藏维度为40

    def forward(self, input, input_h=None):
        """

        :param input: [batch_size, session_num, qd_dim]
        input_h: [batch_size, q_dict_size]
        :return: [batch_size, session_num, hidden_size]
        """

        # 包含输入特征的Tensor
        # 6大行矩阵，没大行为3行10列 10是GRUCell的10
        qd_dim = input.shape[-1]
        session_num = input.shape[-2]
        batch_size = input.shape[-3]
        input = input.permute(1, 0, 2)  # [session_num, batch_size, qd_dim]
        # input = input.reshape(-1, qd_dim)

        # 保存着batch中每个元素的初始化隐状态的Tensor
        # 隐藏层为3行20列；3要和input中的3保持高度一致20为GRUCell的20
        # （说一下，之前的LSTM、LSTMCell、GRU中的隐藏层
        # 和细胞层的行都要高度一致）
        if self.withuser:
            hx = self.proj_q(input_h)
        else:
            hx = torch.zeros(batch_size, self.hidden_size)
            hx = hx.float()

        if CUDA:
            hx = hx.cuda(cudaid)

        # print(hx,"hx1###########")
        # 输出
        output = []
        # 将输出格式保持为（6,3,20）20为要和GRUCell中20要一致
        for i in range(session_num):
            hx = self.gru_cell(input[i], hx)  # 保存着RNN下一个时刻的隐状态。
            output.append(hx)
        # output: [session_num, batch_size, hidden_size)

        output = torch.stack(output)
        output = output.float()
        output = output.permute(1, 0, 2)

        return output, hx


class AttentionNet(torch.nn.Module):
    def __init__(self, q_dict_size=50, hidden_size=50, d_word_vec=100):
        super().__init__()

        self.q_dict_size = q_dict_size  # query_dict size
        self.hidden_size = hidden_size
        self.d_word_vec = d_word_vec
        # self.v = torch.nn.Parameter(torch.rand(self.v_size))
        self.proj = nn.Sequential(nn.Linear(self.q_dict_size, self.hidden_size), nn.ReLU())
        self.proj_v = nn.Linear(self.hidden_size, self.d_word_vec)

    def forward(self, input_x, input_q):
        """atten_net

              Args:
                   input_x(tensor)
                    [batch_size, max_session_num, max_session_length, d_seq_length, d_word_vec]/[batch_size, session_num, d_word_vec]
                    /[batch_size, session_num, sl*dl, d_word_vec]
                    session_qd:[ [batch_size, max_session_num, max_session_length, qd_dim]
                    input_q(tensor):
                    [batch_size, max_session_num, max_session_length,q_dict_size] /[batch_size, q_dict_size]
                    q_atten:
                    [batch_size, max_session_num, q_dict_size]
              Returns:
                 atten_output: [batch_size, max_session_num, max_session_length, d_word_vec]  / [batch_size, d_word_vec]
              """

        weights = self.proj(input_q)
        # [batch_size, max_session_num, max_session_length,hidden_size] /  [batch_size, hidden_size]
        weights = self.proj_v(weights)
        #[batch_size, max_session_num, max_session_length, d_word_vec] /  [batch_size, d_word_vec]
        #[batch_size, max_session_num, qd_dim]
        weights = weights.unsqueeze(-2)
        # [batch_size, max_session_num, max_session_length, 1, d_word_vec] / [batch_size, 1,  d_word_vec]

        weights = weights * input_x
        # [batch_size, max_session_num, max_session_length, d_seq_length, d_word_vec] / [batch_size, session_num, d_word_vec]/
        # [batch_size, session_num, session_length, qd_dim]
        alphas = torch.softmax(weights, dim=-1)
        # [batch_size, max_session_num, max_session_length, d_seq_length] / [batch_size, session_num]
        # [batch_size, session_num, sl * dl]
        atten_output = input_x * alphas
        # [batch_size, max_session_num, max_session_length, d_seq_length, d_word_vec] / [batch_size, session_num, d_word_vec]
        # [batch_size, session_num, sl * dl, d_word_vec]
        atten_output = torch.sum(atten_output, -2)
        # [batch_size, max_session_num, max_session_length, d_word_vec] /  [batch_size, d_word_vec]
        # [batch_size, session_num, d_word_vec]

        return atten_output


class AdditiveAttention(torch.nn.Module):
    def __init__(self, in_dim, v_size=200):
        super().__init__()

        self.in_dim = in_dim
        self.v_size = v_size
        # self.v = torch.nn.Parameter(torch.rand(self.v_size))
        self.proj = nn.Sequential(nn.Linear(self.in_dim, self.v_size), nn.Tanh())
        self.proj_v = nn.Linear(self.v_size, 1)

    def forward(self, context):
        """Additive Attention
        Args:
            context (tensor): [B, seq_len, dim]

        Returns:
            outputs: [B, dim]
        """
        # weights = self.proj(context) @ self.v
        weights = self.proj_v(self.proj(context)).squeeze(-1)
        #  [B, seq_len]
        return torch.bmm(weights.unsqueeze(1), context).squeeze(1)
        # weights.unsqueeze(1):[B, 1, seq_len] * [B, seq_len, dim] = [B, 1, dim]


class QDEncoder(nn.Module):
    def __init__(self,  q_dict_size, d_word_vec):
        super().__init__()
        self.doc_self_attention = SelfAttention(hidden_size=d_word_vec)
        self.atten_net = AttentionNet(q_dict_size=q_dict_size, d_word_vec=d_word_vec)

    def forward(
            self,
            queries, docs
    ):
        """QDEncoder

               Args:
                    queries(tensor):
                    [batch_size, max_session_num, max_session_length, q_dict_size]
                    docs(tensor)
                    [batch_size, max_session_num, max_session_length, d_seq_length, d_word_vec]

               Returns:
                  input (tensor): [batch_size, max_session_num, max_session_length, q_dict_size+d_word_vec]
               """

        embed_dim = docs.shape[-1]
        seq_length = docs.shape[-2]
        docs_shape = docs.shape
        docs = docs.reshape(-1, seq_length, embed_dim)
        doc_encode = self.doc_self_attention(hidden_states=docs)  # [-1, seq_length, embed_dim]
        # query_dim = queries[-1]
        # queries = queries.reshape(-1, query_dim)
        doc_encode = doc_encode.reshape(docs_shape) #...bug?
        doc_encode = self.atten_net(input_x=doc_encode, input_q=queries)
        # queries: [batch_size, max_session_num, max_session_length, q_dict_size]
        # doc_encode:[batch_size, max_session_num, max_session_length, d_word_vec]
        qd_output = torch.cat((queries, doc_encode), dim=-1)

        return qd_output

class QTerm(nn.Module):
    def __init__(self, d_word_vec, q_dict_size):
        super().__init__()
        self.q_dict_size = q_dict_size
        self.doc_self_attention = SelfAttention(hidden_size=d_word_vec)
        self.atten_net = AttentionNet(q_dict_size=q_dict_size, d_word_vec=d_word_vec)


    def forward(
            self,
            q,
            docs
    ):
        """QDEncoder

               Args:
                  :param q: (tensor)
                   [batch_size, q_dict_size]
                  :param docs:(tensor) :
                  [batch_size, max_session_num, max_session_length, d_seq_length, d_word_vec]
              Returns:
              qt_output (tensor): [batch_size, max_session_num, d_word_vec]

               """

        embed_dim = docs.shape[-1]
        seq_length = docs.shape[-2]
        docs_shape = docs.shape
        docs = docs.reshape(-1, seq_length, embed_dim)
        docs = self.doc_self_attention(hidden_states=docs)  # [-1, seq_length, embed_dim]
        docs = docs.reshape(docs_shape) #...bug?

        batch_size = docs.shape[0]
        max_session_num = docs.shape[1]
        d_word_vec = docs.shape[-1]
        docs = docs.reshape(batch_size, max_session_num, -1, d_word_vec)


        q = q.unsqueeze(-2)
        q = q.expand(batch_size, max_session_num, self.q_dict_size)
        # q: [batch_size, max_session_num, q_dict_size]

        qt_output = self.atten_net(input_x=docs, input_q=q)
        #  [batch_size, max_session_num, d_word_vec]



        return  qt_output

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads=5, dropout_prob=0.2):
        """
        假设 hidden_size = 128, num_attention_heads = 8, dropout_prob = 0.2
        即隐层维度为128，注意力头设置为8个
        """
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:  # 整除
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        # 参数定义
        self.num_attention_heads = num_attention_heads  # 8
        self.attention_head_size = int(hidden_size / num_attention_heads)  # 16  每个注意力头的维度 150/5=30
        self.all_head_size = int(self.num_attention_heads * self.attention_head_size)
        # all_head_size = 128 即等于hidden_size, 一般自注意力输入输出前后维度不变

        # query, key, value 的线性变换（上述公式2）
        self.query = nn.Linear(hidden_size, self.all_head_size) # 128, 128
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        # dropout
        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x):
        # INPUT:  x'shape = [bs, seqlen, hid_size]  假设hid_size=128
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # [bs, seqlen, 8, 16]
        x = x.view(*new_x_shape)  #
        return x.permute(0, 2, 1, 3)  # [bs, 8, seqlen, 16]

    def forward(self, hidden_states):
        """

        :param hidden_states: [bs, seqlen, hidden_size]
        :return: [bs, seqlen, hidden_size]
        """
        # eg: attention_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])  shape=[bs, seqlen]
        # attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [bs, 1, 1, seqlen] 增加维度
        # attention_mask = (1.0 - attention_mask) * -10000.0  # padding的token置为-10000，exp(-1w)=0

        # 线性变换
        mixed_query_layer = self.query(hidden_states)  # [bs, seqlen, hid_size]
        mixed_key_layer = self.key(hidden_states)  # [bs, seqlen, hid_size]
        mixed_value_layer = self.value(hidden_states)  # [bs, seqlen, hid_size]

        query_layer = self.transpose_for_scores(mixed_query_layer)  # [bs, heads_num, seqlen, heads_size]
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # 计算query与title之间的点积注意力分数，还不是权重（个人认为权重应该是和为1的概率分布）
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # [bs, 8, seqlen, 16]*[bs, 8, 16, seqlen]  ==> [bs, heads_num, seqlen, seqlen]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # [bs, 8, seqlen, seqlen] math-torch
        # 除以根号注意力头的数量，可看原论文公式，防止分数过大，过大会导致softmax之后非0即1
        # attention_scores = attention_scores + attention_mask
        # 加上mask，将padding所在的表示直接-10000

        # 将注意力转化为概率分布，即注意力权重
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [bs, heads_num, seqlen, seqlen]

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)  # [bs, heads_num, seqlen, seqlen]

        # 矩阵相乘，[bs, heads_num, seqlen, seqlen]*[bs, heads_num, seqlen, heads_size] =[bs, heads_num, seqlen, heads_size]
        context_layer = torch.matmul(attention_probs, value_layer)  # [bs, 8, seqlen, 16]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [bs, seqlen, heads_num, heads_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # [bs, seqlen, hidden_size]
        context_layer = context_layer.reshape(new_context_layer_shape)  # [bs, seqlen, hidden_size] 得到输出
        return context_layer

class DOCP(nn.Module):
    def __init__(self, batch_size, user_dim=40, d_word_vec=100, feature_size=110):
        super().__init__()
        self.d_word_vec = d_word_vec
        self.batch_size = batch_size
        self.q_dict_size = 50
        self.d_seq_length = 100
        self.id_dict_size = 7
        self.qd_dim = self.q_dict_size + d_word_vec
        self.user_dim = user_dim  # user interest vector dim
        self.dict_size = 50 # final score size
        self.feature_size = feature_size

        self.doc_encoder = SelfAttention(hidden_size=128, num_attention_heads=8, dropout_prob=0.2)
        self.qt_module = QTerm(d_word_vec=self.d_word_vec, q_dict_size=self.q_dict_size)
        self.qd_encoder = QDEncoder(q_dict_size=self.q_dict_size, d_word_vec =self.d_word_vec)
        self.embedding = nn.Embedding(len(vocabulary) + 1, self.d_word_vec)
        self.embedding.weight.data.copy_(self.load_embedding())
        # self.additive_atten = AdditiveAttention(in_dim=self.q_dict_size+self.d_word_vec)
        self.additive_atten_d = AdditiveAttention(in_dim=self.d_word_vec)
        self.msa_qd = SelfAttention(hidden_size=self.q_dict_size+self.d_word_vec)
        self.msa_d = SelfAttention(hidden_size=self.d_word_vec)
        self.gru = GRU(input_size=self.qd_dim, hidden_size=self.user_dim)
        # hidden_size: query dict size, embed_dim: in and out put dim
        self.qa_atten_net = AttentionNet(q_dict_size=self.q_dict_size, d_word_vec=self.user_dim)
        self.qd_atten_net = AttentionNet(q_dict_size=self.q_dict_size, d_word_vec=self.qd_dim)

        self.proj_qt_output = nn.Sequential(nn.Linear(self.d_word_vec, self.qd_dim), nn.Sigmoid())
        self.proj_qdprofile_short = nn.Sequential(nn.Linear(self.qd_dim, self.dict_size), nn.Sigmoid())
        self.proj_qtprofile_short = nn.Sequential(nn.Linear(self.d_word_vec, self.dict_size), nn.Sigmoid())
        self.proj_userprofile = nn.Sequential(nn.Linear(self.user_dim, self.dict_size), nn.Sigmoid())
        self.proj_candoc = nn.Sequential(nn.Linear(self.d_word_vec, self.q_dict_size), nn.Sigmoid())
        self.proj_feature = nn.Sequential(nn.Linear(self.feature_size, 1), nn.Tanh())
        self.proj_qwithp = nn.Sequential(nn.Linear(self.q_dict_size+1, self.q_dict_size), nn.Sigmoid())

    def load_embedding(self):  # load the pretrained embedding
        weight = torch.zeros(len(vocabulary) + 1, self.d_word_vec)
        weight[-1] = torch.rand(self.d_word_vec)
        with open('/home/yutong_bai/doc_profile/DocPro/data/word2vec.txt', 'r') as fr:
            for line in fr:
                line = line.strip().split()
                wordid = vocabulary[line[0]]
                weight[wordid, :] = torch.FloatTensor([float(t) for t in line[1:]])
        print("Successfully load the word vectors...")
        return weight

    def pairwise_loss(self, score1, score2):
        return 1 / (1 + torch.exp(torch.sub(score2, score1)))

    def forward(
            self, history_q, history_d, history_qpos, seq, q, d1, d2, y, features1, features2, userid):
        """forward
               Args:
                   history_q (tensor): [batch_size, max_session_num, max_session_length, q_dict_size]
                   history_d (tensor): [batch_size, max_session_num, max_session_length, d_seq_length]
                   history_qpos (tensor): [batch_size, max_session_num, max_session_length, 1]
                   q(tensor):[batch_size, q_dict_size]
                   seq(tensor): [batch_size, 1(query_num)]
                   userid(longtensor):[batch_size, max_id_length]
        """


        batch_size = history_q.shape[0]
        history_q = torch.cat((history_q, history_qpos), dim=-1)
        history_q = self.proj_qwithp(history_q)
        docs = self.embedding(history_d)
        # docs:[batch_size, max_session_num, max_session_length, d_seq_length, d_word_vec]


        # Term-level
        qt_output = self.qt_module(q=q, docs=docs)
        # qt_profile_long: [batch_size, session_num, d_word_vec] qt_profile_short: [batch_size, d_word_vec]
        qd_output = self.qd_encoder(queries=history_q, docs=docs)
        # qd_output:[batch_size, max_session_num, max_session_length, q_dict_size+d_word_vec]
        qt_output =self.proj_qt_output(qt_output)
        # qt_profile_long: [batch_size, max_session_num,  q_dict_size+d_word_vec]
        qt_output = qt_output.unsqueeze(-2)


        # Doc-level
        qd_output = torch.cat((qd_output, qt_output), dim=-2)
        max_session_num = qd_output.shape[-3]
        max_session_length = qd_output.shape[-2]
        qd_dim = qd_output.shape[-1]
        qd_output = qd_output.reshape(-1, max_session_length, qd_dim)
        qd_output = self.msa_qd(qd_output)



        q_atten = q.unsqueeze(-2)
        q_atten = q_atten.expand(batch_size, max_session_num, self.q_dict_size)
        # q: [batch_size, max_session_num, q_dict_size]
        qd_output = qd_output.reshape(batch_size, max_session_num, max_session_length, qd_dim)
        qd_output = self.qd_atten_net(input_x=qd_output, input_q=q_atten)
        qd_output = qd_output.reshape(-1, max_session_num, qd_dim)
        # input: [B, max_session_num, qd_dim]


        # short interest
        qd_output_long, qd_output_short = torch.split(qd_output,[max_session_num-1, 1], dim = -2)
        qd_output_short = qd_output_short.squeeze(-2)
        qd_profile_short = self.proj_qdprofile_short(qd_output_short)

        # long interest
        userid = userid.float()
        gru_output, _ = self.gru(input=qd_output_long)  # gru_output: [batch_size, max_session_num, hidden_size]

        qd_profile_long = self.qa_atten_net(input_x=gru_output, input_q=q)
        qd_profile_long = self.proj_userprofile(qd_profile_long)


        # candidate encoder
        d1 = self.embedding(d1)  # [batch_size, seq_length,embed_dim]
        d1 = self.msa_d(d1)
        d1 = self.additive_atten_d(d1)  # [batch_size, embed_dim]
        d1 = self.proj_candoc(d1)

        d2 = self.embedding(d2)  # [batch_size, seq_length,embed_dim]
        d2 = self.msa_d(d2)
        d2 = self.additive_atten_d(d2)  # [batch_size, embed_dim]
        d2 = self.proj_candoc(d2)  # [batch_size, dict_size]

        l_user_mo = torch.sqrt(torch.sum(torch.square(qd_profile_long), -1))
        s_user_mo = torch.sqrt(torch.sum(torch.square(qd_profile_short), -1))
        d1_mo = torch.sqrt(torch.sum(torch.square(d1), -1))
        d2_mo = torch.sqrt(torch.sum(torch.square(d2), -1))

        score_l_1 = torch.unsqueeze((torch.sum(qd_profile_long * d1, -1)) / (l_user_mo * d1_mo + 1), -1)
        score_s_1 = torch.unsqueeze((torch.sum(qd_profile_short * d1, -1)) / (s_user_mo * d1_mo + 1), -1)
        features1 = self.proj_feature(features1) # [batch_size,1]
        score_1 = score_l_1 + score_s_1 + features1
        # [batch_size, 1] +1?

        score_l_2 = torch.unsqueeze((torch.sum(qd_profile_long * d2, -1)) / (l_user_mo * d2_mo + 1), -1)
        score_s_2 = torch.unsqueeze((torch.sum(qd_profile_short * d2, -1)) / (s_user_mo * d2_mo + 1), -1)
        features2 = self.proj_feature(features2) # [batch_size,1]
        score_2 = score_l_2 + score_s_2 + features2


        score = torch.cat((score_1, score_2), 1)  # [batch_size, 2]
        p1_score = self.pairwise_loss(score_1, score_2)
        p2_score = self.pairwise_loss(score_2, score_1)
        p_score = torch.cat((p1_score, p2_score), 1) # [batch_size, 2]

        preds = torch.softmax(score, dim=-1)
        preds = torch.argmax(preds, 1)
        preds = preds.long()
        correct = preds.eq(y)
        correct = correct.float()
        accuracy = torch.mean(correct)

        # loss = torch.sum(lambdas * F.cross_entropy(input=p_score, target=y))  # y 0?

        # dim = ?
        return score, p_score, accuracy




# def main():
#     datasource = DataSet(max_query=args.max_query, limitation=args.limitation,
#                          batch_size=args.batch_size, num_epoch=args.num_epoch)
#     datasource.prepare_hierarchical_dataset()
#     model = DOCP()
#     model = model.train_network(datasource)
#
#     # evaluation = AP()
#     # print("Train Losses in different epoches are ", model.train_losses)
#     # print("Train Accuracy in different epoches are ", model.train_accuracy)
#     # print("Valid Losses in different epoches are ", model.valid_losses)
#     # print("Valid Accuracy in different epoches are ", model.valid_accuracy)
#     # if model.test_model:
#     # 	model = model.test_network(datasource)
#     if model.score_model:
#         datasource.prepare_score_dataset()
#         model.score_network(datasource, evaluation)
#
#
# if __name__ == '__main__':
#     main()

import os
import pickle
import itertools
import numpy as np
import torch
from torch import nn
import torch.utils.data
from time import time
from docpro import DOCP
from metric import AP, MRR
from datetime import datetime

import pdb



os.environ['CUDA_VISIBLE_DEVICES']='0, 1'

vocab = pickle.load(open('/home/yutong_bai/doc_profile/DocPro/data/vocab.dict', 'rb'))
batch_size = 100
dict_size = 50
q_dict_size = 50
d_seq_length = 50
max_query_num = 105
max_id_length = 7
max_id_length = 7
cudaid = 1

# pid 数字


def divide_session(history_x, max_session_length=20, max_session_num=4):
    """
    :param history_x: [item_num, query_num, q_dict_size+d_seq_length+2]
    :param max_session_length:
    :return: history_q(with padding): [item_num, max_session_num, max_session_length, q_dict_size]
    history_d(with padding): [item_num, max_session_num, max_session_length, b_seq_length]
        history_qpos :[item_num, max_session_num, max_session_length, 1]
    """
    # qd_dim = input.shape[-1]
    # history = torch.cat((qd_pair, b), dim=-1)
    # history = history.reshape(-1, qd_dim + 1)
    # for history_i in history:
    #     bi = history_i[-1]
    #     qd_pair, b = torch.split(history_i, split_size_or_sections=[int(history_x.get_shape()[1] - 1), 1], dim=

    query_count = 0
    query_in_session_count = 0
    session_q = []
    session_d = []
    history_item_q = []  # divided session for one item
    history_item_d = []
    history_seq_pad = []
    history = []
    history_q = []
    history_d = []
    session_pad_q = []
    session_pad_d = []
    history_qpos =[]
    history_qpos_item =[]
    history_qpos_insession =[]

    seq_pad = np.zeros(q_dict_size + d_seq_length + 2)
    for i in range(0, max_session_length):
        session_pad_q.append(seq_pad[0:q_dict_size])
        session_pad_d.append(seq_pad[0:d_seq_length])


    query_sum = 0
    for item in history_x:
        query_num = len(item)
        query_sum += query_num

    ave_query_num = query_sum/len(history_x)
    query_in_session_count_sum = 0
    session_num_sum = 0


    for item in history_x:
        b = []
        query_num = len(item)
        if query_num < max_query_num:
            for i in range(0, max_query_num - query_num):
                item.append(seq_pad)
        else:
            item = item[0:max_query_num]
        for history_seq in item:
            bi = history_seq[-2]
            b.append(bi)
            # new session
            if bi == 1:
                if query_count != 0:
                    session_q = session_q[0:max_session_length]
                    session_d = session_d[0:max_session_length]
                    # padding
                    if query_in_session_count < max_session_length:
                        for i in range(0, max_session_length - query_in_session_count):
                            session_q.append(seq_pad[0:q_dict_size])
                            session_d.append(seq_pad[0:d_seq_length])  # ....?
                    history_item_q.append(session_q)
                    history_item_d.append(session_d)
                    session_q = []
                    session_d = []
                    query_in_session_count_sum += query_in_session_count
                    query_in_session_count = 0
                    session_q.append(history_seq[0:q_dict_size])
                    session_d.append(history_seq[q_dict_size:q_dict_size + d_seq_length])
                    query_in_session_count += 1
                else:
                    session_q.append(history_seq[0:q_dict_size])
                    session_d.append(history_seq[q_dict_size:q_dict_size + d_seq_length])
                    query_in_session_count += 1

            else:
                session_q.append(history_seq[0:q_dict_size])
                session_d.append(history_seq[q_dict_size:q_dict_size + d_seq_length])
                query_in_session_count += 1
            query_count += 1

        query_count = 0
        # the last session:
        if session_q:
            session_q = session_q[0:max_session_length]
            session_d = session_d[0:max_session_length]
            # padding
            if query_in_session_count < max_session_length:
                for i in range(0, max_session_length - query_in_session_count):
                    session_q.append(seq_pad[0:q_dict_size])
                    session_d.append(seq_pad[0:d_seq_length])
            history_item_q.append(session_q)
            history_item_d.append(session_d)
            session_q = []
            session_d = []
            query_in_session_count = 0

        session_num = len(history_item_q)
        session_num_sum += session_num
        history_item_q = history_item_q[0:max_session_num]
        history_item_d = history_item_d[0:max_session_num]
        if session_num < max_session_num:
            for i in range(0, max_session_num - session_num):
                history_item_q.append(session_pad_q)
                history_item_d.append(session_pad_d)
        history_q.append(history_item_q)
        history_d.append(history_item_d)
        history_item_q = []
        history_item_d = []

    ave_session_num = session_num_sum/len(history_x)
    ave_session_length = query_in_session_count_sum/session_num_sum
    # history_q(with padding): [item_num, max_session_num, max_session_length, q_dict_size]
    # history_d(with padding): [item_num, max_session_num, max_session_length, b_seq_length]

    for item in history_q:
        for session in item:
            i = 0
            for query in session:
                history_qpos_insession.append([i])
                i = i+1
            history_qpos_item.append(history_qpos_insession)
            history_qpos_insession = []
        history_qpos.append(history_qpos_item)
        history_qpos_item = []

    return history_q, history_d, history_qpos


def collate_fn_test(insts):
    ''' Pad the instance to the max seq length in batch '''
    history_q_test, history_d_test, history_qpos_test, seq_test, d1_test, Y_test, q_test, \
    f1_test, user_test, lines_test = zip(
        *insts)


    history_q_test = torch.FloatTensor(history_q_test)
    history_d_test = torch.LongTensor(history_d_test)
    history_qpos_test = torch.FloatTensor(history_qpos_test)
    seq_test = torch.LongTensor(seq_test)
    d1_test = torch.LongTensor(d1_test)
    Y_test = torch.LongTensor(Y_test)
    q_test = torch.FloatTensor(q_test)
    f1_test = torch.FloatTensor(f1_test)
    user_test = torch.LongTensor(user_test)

    return history_q_test, history_d_test,history_qpos_test, seq_test, d1_test, Y_test, q_test, f1_test,  \
           user_test, lines_test


def collate_fn_train(insts):
    ''' Pad the instance to the max seq length in batch '''
    history_q_train, history_d_train, history_qpos_train, seq_train, d1_train, d2_train, Y_train, q_train, lambda_train, \
    f1_train, f2_train, user_train = zip(
        *insts)

    # history_train = []
    # for item in X_train:
    #     history_train.append(torch.LongTensor(item))

    # history_q_train = padding_batch(history_q_train, dict_size)
    # history_q_train = torch.FloatTensor(history_q_train)
    #
    # history_d_train = padding_batch(history_d_train, dict_size)
    # history_d_train = torch.LongTensor(history_d_train)
    #
    # b_train = padding_batch_int(b_train, 1)
    # b_train = torch.LongTensor(b_train)
    #
    # e_train = padding_batch_int(e_train, 1)
    # e_train = torch.LongTensor(e_train)
    history_q_train = torch.FloatTensor(history_q_train)
    history_qpos_train = torch.FloatTensor(history_qpos_train)
    history_d_train = torch.LongTensor(history_d_train)
    seq_train = torch.LongTensor(seq_train)
    d1_train = torch.LongTensor(d1_train)
    d2_train = torch.LongTensor(d2_train)
    Y_train = torch.LongTensor(Y_train)
    q_train = torch.FloatTensor(q_train)
    lambda_train = torch.FloatTensor(lambda_train)
    f1_train = torch.FloatTensor(f1_train)
    f2_train = torch.FloatTensor(f2_train)
    user_train = torch.LongTensor(user_train)

    return history_q_train, history_d_train, history_qpos_train, seq_train, d1_train, d2_train, Y_train, q_train, lambda_train, f1_train, \
           f2_train, user_train


class Dataset_train(torch.utils.data.Dataset):
    def __init__(
            self, history_q_train, history_d_train, history_qpos_train, seq_train, d1_train, d2_train, Y_train, q_train,
            lambda_train, f1_train, f2_train, user_train):
        self.history_q_train = history_q_train
        self.history_d_train = history_d_train
        self.history_qpos_train = history_qpos_train
        self.seq_train = seq_train
        self.d1_train = d1_train
        self.d2_train = d2_train
        self.Y_train = Y_train
        self.q_train = q_train
        self.lambda_train = lambda_train
        self.f1_train = f1_train
        self.f2_train = f2_train
        self.user_train = user_train

    def __len__(self):
        return len(self.history_q_train)

    def __getitem__(self, idx):
        history_q_train = self.history_q_train[idx]
        history_d_train = self.history_d_train[idx]
        history_qpos_train = self.history_qpos_train[idx]
        seq_train = self.seq_train[idx]
        d1_train = self.d1_train[idx]
        d2_train = self.d2_train[idx]
        Y_train = self.Y_train[idx]
        q_train = self.q_train[idx]
        lambda_train = self.lambda_train[idx]
        f1_train = self.f1_train[idx]
        f2_train = self.f2_train[idx]
        user_train = self.user_train[idx]

        return history_q_train, history_d_train, history_qpos_train, seq_train, d1_train, d2_train, Y_train, q_train, \
               lambda_train, f1_train, f2_train, user_train


class Dataset_test(torch.utils.data.Dataset):
    def __init__(
            self, history_q_test, history_d_test, history_qpos_test, seq_test, d1_test, Y_test, q_test,
             f1_test, user_test, lines_test):
        self.history_q_test = history_q_test
        self.history_d_test = history_d_test
        self.history_qpos_test = history_qpos_test
        self.seq_test = seq_test
        self.d1_test = d1_test
        self.Y_test = Y_test
        self.q_test = q_test
        self.f1_test = f1_test
        self.user_test = user_test
        self.lines_test = lines_test

    def __len__(self):
        return len(self.history_q_test)

    def __getitem__(self, idx):
        history_q_test = self.history_q_test[idx]
        history_d_test = self.history_d_test[idx]
        history_qpos_test = self.history_qpos_test[idx]
        seq_test = self.seq_test[idx]
        d1_test = self.d1_test[idx]
        Y_test = self.Y_test[idx]
        q_test = self.q_test[idx]
        f1_test = self.f1_test[idx]
        user_test = self.user_test[idx]
        lines_test = self.lines_test[idx]

        return history_q_test, history_d_test, history_qpos_test, seq_test, d1_test, Y_test, q_test, \
               f1_test, user_test, lines_test


class DataSet:
    def __init__(self, max_query=300, limitation=10000000,
                 batch_size=200, num_epoch=20, min_session=4,
                 data_path='/home/yutong_bai/doc_profile/DocPro/data/',
                 demo_or_not=False):

        self.data_path = data_path
        self.in_path = os.path.join(data_path, 'HF/')
        self.filenames = sorted(os.listdir(self.in_path))
        # self.filenames = sorted(os.listdir('/home/yutong_bai/doc_profile/DocPro/data/test/'))
        # self.filenames = sorted(os.listdir('/home/yutong_bai/doc_profile/DocPro/data/HF/'))
        self.query2vec_path = os.path.join(data_path, 'QueryVec/')
        # self.doc2vec_path = os.path.join(data_path, 'DocVec/')
        # self.dict_size = 50  # Length of Query2Vec/Doc2Vec embedding


        self.max_doclen = 30  # the max number of words in a doc title
        self.feature_size = 110
        self.limitation = limitation
        self.max_query = max_query
        self.min_session = min_session  # 滤掉session少于一定数目的用户
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.cutoff_date = '2006-04-03 00:00:00'  # before, history dataset; after experiment dataset
        self.train_date = '2006-05-16 00:00:00'  # before, train dataset; after, valid/test dataset
        # self.cutoff_date = datetime.strptime('2/12/2013', '%m/%d/%Y')
        self.demo = demo_or_not

    def init_dict(self):
        self.query2id = pickle.load(open(os.path.join(self.data_path, 'query2id.dict'), 'rb'))  # len: 1569274
        self.url2id = pickle.load(open(os.path.join(self.data_path, 'url2id2.dict'), 'rb'))  # len: 626480

    def transform_q(self, sentence):
        idx = self.query2id[sentence]
        return pickle.load(open(os.path.join(self.query2vec_path, str(idx) + '.pkl'), 'rb'))

    def transform_userid(self, id):
        ids = []

        for snum in id:
            num = ord(snum)
            ids.append(num)

        ids = ids[0:max_id_length]
        if len(id) < max_id_length:
            ids += [0] * (max_id_length - len(id))

        return ids



    # def transform_u(self, sentence):
    #     idx = self.url2id[sentence]
    #     return pickle.load(open(os.path.join(self.doc2vec_path, str(idx) + '.pkl'), 'rb'))

    def sen2did(self, sen):
        """
      Args:
         sen(word)

      Returns:
         word id
      """
        idx = []
        for word in sen.split():
            if word in vocab:
                idx.append(vocab[word])
            else:
                idx.append(vocab['<unk>'])
        idx = idx[:d_seq_length]
        padding = [0] * (d_seq_length - len(idx))
        idx = idx + padding
        return idx

    def init_dataset(self):
        self.X_train = []
        self.history_q_train = []
        self.history_d_train = []
        self.history_x_train = []  # session
        self.history_qpos_train =[]
        self.seq_train = []
        self.d1_train = []
        self.d2_train = []
        self.Y_train = []
        self.q_train = []
        self.lambda_train = []
        self.features1_train = []
        self.features2_train = []
        self.X_valid = []
        self.seq_valid = []
        self.d1_valid = []
        self.d2_valid = []
        self.Y_valid = []
        self.q_valid = []
        self.lambda_valid = []
        self.features1_valid = []
        self.features2_valid = []
        self.X_test = []
        self.history_q_test = []
        self.history_d_test = []
        self.history_qpos_test =[]
        self.seq_test = []
        self.d1_test = []
        self.d2_test = []
        self.Y_test = []
        self.q_test = []
        self.lambda_test = []
        # self.lines_train = [] # no use
        self.lines_test = []
        self.features1_test = []
        self.features2_test = []
        self.user_train = []
        self.user_valid = []
        self.user_test = []
        self.time_train = []
        self.time_valid = []
        self.time_test = []
        self.X_tmp = []
        self.seq_tmp = []
        self.q_tmp = []
        self.user_tmp = []
        self.lines_tmp = []  # for debugging

    def divide_dataset(self, filename):
        '''
        count session in experiment dataset
        return: True if #session >= min_session else False
        '''
        session_sum = 0
        # session_train = 0
        # session_valid = 0
        # session_test = 0
        query_sum = 0
        last_queryid = 0
        last_sessionid = 0

        with open(os.path.join(self.in_path, filename)) as fhand:
            for line in fhand:
                try:
                    line, features = line.rstrip().split('###')
                except:
                    line = line.rstrip()
                user, sessionid, querytime, query, url, title, sat, urlrank = line.strip().split('\t')
                queryid = sessionid + querytime + query

                if querytime < self.cutoff_date:
                    if queryid != last_queryid:
                        last_queryid = queryid
                    query_sum += 1
                elif querytime < self.train_date:
                    if query_sum < 2:
                        return False
                    if sessionid != last_sessionid:
                        session_sum += 1
                        # session_train += 1
                        assert (last_queryid != queryid)
                        last_sessionid = sessionid
                    if queryid != last_queryid:
                        last_queryid = queryid
                else:
                    if sessionid != last_sessionid:  # 这里不区分valid 和 test
                        session_sum += 1
                        # session_valid += 1
                        assert (last_queryid != queryid)
                        last_sessionid = sessionid
                    if queryid != last_queryid:
                        last_queryid = queryid

            if session_sum < self.min_session:
                return False
            # session_test = session_valid
            # session_valid //= 2
            # session_test -= session_valid
        return True

    def cal_delta(self, targets):
        '''cal lambda in lambdaRank algorithm'''
        n_targets = len(targets)
        deltas = np.zeros((n_targets, n_targets))
        total_num_rel = 0
        total_metric = 0.0
        for i in range(n_targets):
            if targets[i] == '1':
                total_num_rel += 1
                total_metric += total_num_rel / (i + 1.0)
        metric = (total_metric / total_num_rel) if total_num_rel > 0 else 0.0
        num_rel_i = 0
        for i in range(n_targets):
            if targets[i] == '1':
                num_rel_i += 1
                num_rel_j = num_rel_i
                sub = num_rel_i / (i + 1.0)
                for j in range(i + 1, n_targets):
                    if targets[j] == '1':
                        num_rel_j += 1
                        sub += 1 / (j + 1.0)
                    else:
                        add = (num_rel_j / (j + 1.0))
                        new_total_metric = total_metric + add - sub
                        new_metric = new_total_metric / total_num_rel
                        deltas[i, j] = new_metric - metric

            else:
                num_rel_j = num_rel_i
                add = (num_rel_i + 1) / (i + 1.0)

                for j in range(i + 1, n_targets):
                    if targets[j] == '1':
                        sub = (num_rel_j + 1) / (j + 1.0)
                        new_total_metric = total_metric + add - sub
                        new_metric = new_total_metric / total_num_rel
                        deltas[i, j] = new_metric - metric
                        num_rel_j += 1
                        add += 1 / (j + 1.0)

        return deltas

    def prepare_pairdata(self, query_count, sat_list, feature_list, doc_list, history_seq, queryvec, flag, user_id,
                         qtime):
        '''将一个query及其对应的文档列表转换成一系列的文档对'''
        user_id = self.transform_userid(user_id)
        delta = self.cal_delta(sat_list)
        n_targets = len(sat_list)
        for i in range(n_targets):
            for j in range(i + 1, n_targets):  # 只考虑有意义的pair
                if delta[i, j] > 0:
                    rel_vec = doc_list[j]
                    rel_features = feature_list[j]
                    irr_vec = doc_list[i]
                    irr_features = feature_list[i]
                    lbd = delta[i, j]
                elif delta[i, j] < 0:
                    rel_vec = doc_list[i]
                    rel_features = feature_list[i]
                    irr_vec = doc_list[j]
                    irr_features = feature_list[j]
                    lbd = -delta[i, j]
                else:  # 无意义
                    continue
                # if session_count <= self.session_train:
                if flag == 'train':  # 直接用flag来区分不同的数据集
                    self.X_train.append(history_seq[-self.max_query:query_count])
                    # query_count 指向前一个query--history中query个数
                    self.seq_train.append((query_count) if query_count < 300 else 299)
                    self.d1_train.append(rel_vec)
                    self.d2_train.append(irr_vec)
                    self.Y_train.append(0)
                    self.q_train.append(queryvec)
                    self.lambda_train.append(lbd)
                    self.features1_train.append(rel_features)
                    self.features2_train.append(irr_features)
                    self.user_train.append(user_id)
                    self.time_train.append(qtime)
                elif flag == 'valid':
                    # elif session_count <= self.session_train+self.session_valid:
                    self.X_valid.append(history_seq[-self.max_query:query_count])
                    self.seq_valid.append(query_count if query_count < 300 else 299)
                    self.d1_valid.append(rel_vec)
                    self.d2_valid.append(irr_vec)
                    self.Y_valid.append(0)
                    self.q_valid.append(queryvec)
                    self.lambda_valid.append(lbd)
                    self.features1_valid.append(rel_features)
                    self.features2_valid.append(irr_features)
                    self.user_valid.append(user_id)
                    self.time_valid.append(qtime)
                # else:
                #     self.X_test.append(history_seq[-self.max_query:query_count])
                #     self.seq_test.append(query_count if query_count < 300 else 299)
                #     self.d1_test.append(rel_vec)
                #     self.d2_test.append(irr_vec)
                #     self.Y_test.append(0)
                #     self.q_test.append(queryvec)
                #     self.lambda_test.append(lbd)
                #     self.features1_test.append(rel_features)
                #     self.features2_test.append(irr_features)
                #     self.user_test.append(user_id)
                #     self.time_test.append(qtime)

    #
    # def prepare_score_dataset(self):
    #     self.init_dataset()
    #     if not hasattr(self, 'query2id'):
    #         self.init_dict()
    #     for filename in self.filenames:
    #         if not self.divide_dataset(filename):
    #             continue
    #         query_count = -1
    #         history_seq = []
    #         last_queryid = 0
    #         last_sessionid = 0
    #         session_count_all = 0
    #         session_count = 0
    #         clicked_url_init = np.zeros((self.dict_size), dtype=np.float64)
    #         fhand = open(os.path.join(self.in_path, filename))
    #         for line in fhand:
    #             try:
    #                 line, features = line.rstrip().split('###')
    #                 user, sessionid, querytime, query, url, title, sat, urlrank = line.split('\t')
    #                 features = [float(item) for item in features.split('\t')]
    #             except:
    #                 user, sessionid, querytime, query, url, title, sat, urlrank = line.split('\t')
    #             # current_date = datetime.strptime(date, '%m/%d/%Y')
    #             dids = self.sen2did(title)
    #             if sessionid != last_sessionid:
    #                 session_count_all += 1
    #             if current_date < self.cutoff_date:
    #                 if queryid != last_queryid:
    #                     queryvec = self.transform_q(query.lower())
    #                     query_count += 1
    #                     if sessionid != last_sessionid:
    #                         history_seq.append(np.append([queryvec, clicked_url_init], [1.0, 0.0]))
    #                         if last_queryid != 0:
    #                             history_seq[query_count - 1][-1] = 1.0
    #                         last_sessionid = sessionid
    #                     else:
    #                         history_seq.append(np.append([queryvec, clicked_url_init], [0.0, 0.0]))
    #                 if int(sat) == 1:
    #                     history_seq[query_count][self.dict_size:2 * self.dict_size] += self.transform_u(url.lower())
    #             else:
    #                 docvec = self.transform_u(url.lower())
    #                 queryvec = self.transform_q(query.lower())
    #                 if queryid != last_queryid:
    #                     if sessionid != last_sessionid:
    #                         history_seq.append(np.append([queryvec, clicked_url_init], [1.0, 0.0]))
    #                         if last_queryid != 0:
    #                             history_seq[query_count - 1][-1] = 1.0
    #                     else:
    #                         history_seq.append(np.append([queryvec, clicked_url_init], [0.0, 0.0]))
    #                     last_queryid = queryid
    #                     query_count += 1
    #                 if sessionid != last_sessionid:
    #                     session_count += 1
    #                     last_sessionid = sessionid
    #                 if int(sat) == 1:
    #                     history_seq[query_count][self.dict_size:2 * self.dict_size] += self.transform_u(url.lower())
    #                 if session_count > self.session_train + self.session_valid:
    #                     self.X_test.append(history_seq[-self.max_query:query_count])
    #                     self.seq_test.append(query_count if query_count < 300 else 299)
    #                     self.d1_test.append(docvec)
    #                     self.q_test.append(queryvec)
    #                     self.features1_test.append(features)
    #                     self.lines_test.append(line + '\t' + str(session_count_all))
    #     print("Successfully prepared the test set.")
    #
    def prepare_hierarchical_dataset(self):
        if not hasattr(self, 'X_train'):
            self.init_dataset()
        if not hasattr(self, 'query2id'):
            self.init_dict()
        key = 0
        for count, filename in enumerate(self.filenames):
            # print("count:", count, "filename:", filename)
            if count % 1000 == 999:
                print("processing %d/%d" % (count + 1, len(self.filenames)))
            if not self.divide_dataset(filename):  # Judge valid user， 判断这个用户的数据是否有效
                print(filename, 'not valid')
                continue
            # pdb.set_trace()
            if key == 1:  # Query in last session
                self.prepare_pairdata(query_count, sat_list, feature_list,
                                      doc_list, history_seq, self.transform_q(query), flag, user, querytime)
            key = 0  # 用来标识当前查询下是否有点击文档，若有则构成文本对训练样本
            query_count = -1  # Query_count point to the present query in query sequence.
            history_seq = []
            # hs = []
            doc_list = []
            sat_list = []
            feature_list = []
            flag = ''
            titles = ''
            last_queryid = 0
            last_sessionid = 0
            last_querytime = 0
            last_query = ''
            clicked_url_init = np.zeros((d_seq_length), dtype=np.float64)  # doc embedding？
            # holder = ''
            fcontent = open(os.path.join(self.in_path, filename), 'r').readlines()
            lno = 0
            for line in fcontent:
                lno += 1
                try:
                    line, features = line.rstrip().split('###')
                    features = [float(item) for item in features.split('\t')]
                    if np.isnan(np.array(features)).sum():  # 除去包含nan的feature..？
                        continue
                except:
                    line = line.rstrip()

                user, sessionid, querytime, query, url, title, sat, urlrank = line.split('\t')
                queryid = sessionid + '_' + querytime + '_' + query
                did = self.sen2did(title)



                if querytime < self.cutoff_date:
                    if queryid != last_queryid:
                        queryvec = self.transform_q(query)
                        titles = ''
                        if sessionid != last_sessionid:
                            qd = np.concatenate((queryvec, clicked_url_init), axis=0)
                            history_seq.append(np.append(qd, [1.0, 0.0]))
                            # hs.append([user, lno, queryid, holder, 1.0, 0.0])
                            # history_seq:[query_vector, url, 0.0, 0.0]
                            # the first in a session [query_vector, url, 1.0, 0.0]
                            # history_seq shape:[query_num, X]
                            if last_queryid != 0:
                                history_seq[query_count][-1] = 1.0
                            last_sessionid = sessionid
                        else:  # 最后一个session
                            qd = np.concatenate((queryvec, clicked_url_init), axis=0)
                            history_seq.append(np.append(qd, [0.0, 0.0]))
                            # hs.append([user, lno, queryid, holder, 0.0, 0.0])
                        query_count += 1
                        last_queryid = queryid
                        last_querytime = querytime
                        last_query = query
                    if int(sat) == 1:  # 只有SAT的文档会转化为vector
                        titles += ' ' + title
                        dids = self.sen2did(titles)
                        history_seq[query_count][q_dict_size:q_dict_size+d_seq_length] = dids
                        # -- the last docs
                        # hs[query_count][3] = url
                        # key = 1 # add or not are the same, since only sat == 1 is in the history, can't form pairdata d
                else: # test data
                    if last_querytime!= 0 :
                        if last_querytime> self.train_date:
                            self.X_test.append(history_seq[-self.max_query:query_count])
                            self.seq_test.append(query_count if query_count < 300 else 299)
                            self.d1_test.append(did)
                            self.q_test.append(self.transform_q(last_query))
                            self.features1_test.append(features)
                            self.lines_test.append(line)
                            self.Y_test.append(0)
                            self.user_test.append(self.transform_userid(user))
                    if queryid != last_queryid:  # 一个query的记录全部访问结束时基于这个(之前一个）query生成pairdate
                        titles = ''
                        if key == 1:  # There is a SAT-click in the session
                            # if querytime < self.train_date: # There was a bug, judge ought to based on last querytime, since last interaction pairdata to be prepared
                            if last_querytime < self.train_date:
                                flag = 'train'
                            elif len(sat_list) < 10:  # == 5
                                flag = 'valid'
                            else:
                                flag = 'test'


                            # pdb.set_trace()
                            self.prepare_pairdata(query_count, sat_list, feature_list,
                                                  doc_list, history_seq, self.transform_q(last_query), flag, user,
                                                  last_querytime)  # note that it's last_query
                            key = 0
                        doc_list = []
                        sat_list = []
                        feature_list = []
                        queryvec = self.transform_q(query)
                        if sessionid != last_sessionid:
                            last_sessionid = sessionid
                            qd = np.concatenate((queryvec, clicked_url_init), axis=0)
                            history_seq.append(np.append(qd, [1.0, 0.0]))
                            # hs.append([user, lno, queryid, holder, 1.0, 0.0])
                            # [queryvec1, clicked_url_init, 1.0, 1.0]
                            # [queryvec2, url, , ]
                            # [queryvec3, url, , ]
                            # query count: index of query_id

                            if last_queryid != 0:
                                history_seq[query_count][-1] = 1.0
                                # hs[query_count][-1] = 1.0
                        else:
                            qd = np.concatenate((queryvec, clicked_url_init), axis=0)
                            history_seq.append(np.append(qd, [0.0, 0.0]))
                            # hs.append([user, lno, queryid, holder, 0.0, 0.0])
                        query_count += 1
                        last_queryid = queryid
                        last_querytime = querytime
                        last_query = query
                    doc_list.append(did)
                    sat_list.append(sat)
                    feature_list.append(features)
                    if int(sat) == 1:
                        titles += ' ' + title
                        dids = self.sen2did(titles)
                        history_seq[query_count][q_dict_size:q_dict_size+d_seq_length] = dids
                        # hs[query_count][3] = url
                        key = 1
                    if len(self.Y_train) > self.limitation:
                        print('out of limitation in userlog:', filename)
                        break
            # pdb.set_trace()
            if self.demo and len(self.Y_train) > 3000:
                print("Successfully prepared the train:{}, valid:{} and test:{} set for SHORT."
                      .format(len(self.Y_train), len(self.Y_valid), len(self.Y_test)))
                break

        self.history_q_train, self.history_d_train, self.history_qpos_train = divide_session(history_x=self.X_train)
        self.history_q_test, self.history_d_test, self.history_qpos_test = divide_session(history_x=self.X_test)
        print("Successfully prepared the train:{}, valid:{} and test:{} set."
              .format(len(self.Y_train), len(self.Y_valid), len(self.Y_test)))


model = DOCP(batch_size=batch_size)
model = model.cuda(cudaid)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


def test(test_loader, load=False):
    if load == True:
        model.load_state_dict(torch.load('model.b'))
    model.eval()
    test_loss = 0
    correct = 0
    f = open('test_score.txt', 'w')
    for history_q_test, history_d_test, history_qpos_test, seq_test, d1_test, Y_test, q_test, \
        f1_test, user_test, lines_test in test_loader:

        history_q_test = history_q_test.cuda(cudaid)
        history_d_test = history_d_test.cuda(cudaid)
        history_qpos_test = history_qpos_test.cuda(cudaid)

        seq_test = seq_test.cuda(cudaid)
        d1_test = d1_test.cuda(cudaid)
        Y_test = Y_test.cuda(cudaid)
        q_test = q_test.cuda(cudaid)
        f1_test = f1_test.cuda(cudaid)
        user_test = user_test.cuda(cudaid)

        score, p_score, accuracy = model(history_q=history_q_test, history_d=history_d_test,
                                         history_qpos =history_qpos_test, seq=seq_test, q=q_test,
                                         d1=d1_test, d2=d1_test, y=Y_test, features1=f1_test, features2=f1_test,
                                         userid=user_test)

        assert (len(score) == len(lines_test))
        for line, sc in zip(lines_test, score):
            f.write(line + '\t' + str(float(sc[0])) + '\n')


def train(train_loader):
    model.train()
    loss_sum = 0
    batch_num = 0
    for batch_idx, (
            history_q_train, history_d_train, history_qpos_train, seq_train, d1_train, d2_train, Y_train, q_train,
            lambda_train, f1_train, f2_train, user_train) in enumerate(
        train_loader):
        optimizer.zero_grad()

        history_q_train = history_q_train.cuda(cudaid)
        history_d_train = history_d_train.cuda(cudaid)
        history_qpos_train = history_qpos_train.cuda(cudaid)

        seq_train = seq_train.cuda(cudaid)
        d1_train = d1_train.cuda(cudaid)
        d2_train = d2_train.cuda(cudaid)
        Y_train = Y_train.cuda(cudaid)
        q_train = q_train.cuda(cudaid)
        lambda_train = lambda_train.cuda(cudaid)
        f1_train = f1_train.cuda(cudaid)
        f2_train = f2_train.cuda(cudaid)
        user_train = user_train.cuda(cudaid)

        score, p_score, accuracy = model(history_q=history_q_train, history_d=history_d_train,
                                         history_qpos =history_qpos_train, seq=seq_train, q=q_train,
                                         d1=d1_train, d2=d2_train, y=Y_train, features1=f1_train, features2=f2_train,
                                        userid=user_train)

        loss = torch.sum(lambda_train * criterion(p_score, Y_train))
        loss = loss.requires_grad_()
        if batch_idx == 0:
            loss.backward(retain_graph=True)
        else:
            loss.backward()
        optimizer.step()
        print("---batch:", batch_idx, "---------------")
        print(loss)
        print("accuracy:", accuracy)

        loss_sum += loss
        batch_num = batch_idx
    torch.save(model.state_dict(), 'model.b')
    loss_ave= loss_sum/(batch_num+1)
    return loss_ave


if __name__ == "__main__":
    data = DataSet(demo_or_not=True, batch_size=batch_size)
    data.prepare_hierarchical_dataset()
    torch_Dataset_train = Dataset_train(
        history_q_train=data.history_q_train, history_d_train=data.history_d_train,
        history_qpos_train=data.history_qpos_train, seq_train=data.seq_train,
        d1_train=data.d1_train, d2_train=data.d2_train,
        Y_train=data.Y_train, q_train=data.q_train, lambda_train=data.lambda_train, f1_train=data.features1_train,
        f2_train=data.features2_train, user_train=data.user_train)
    train_loader = torch.utils.data.DataLoader(
        torch_Dataset_train,
        batch_size=data.batch_size,
        collate_fn=collate_fn_train)

    torch_Dataset_test = Dataset_test(
        history_q_test=data.history_q_test, history_d_test=data.history_d_test,
        history_qpos_test=data.history_qpos_test,seq_test=data.seq_test,
        d1_test=data.d1_test,
        Y_test=data.Y_test, q_test=data.q_test,  f1_test=data.features1_test,
        user_test=data.user_test, lines_test=data.lines_test)

    test_loader = torch.utils.data.DataLoader(
        torch_Dataset_test,
        batch_size=64,
        collate_fn=collate_fn_test)

    evaluation = AP()
    model.load_state_dict(torch.load('model.b'))


    re = open('results.txt', 'a')
    ave_loss = 0
    count = 0
    for epoch in range(1):
        print("---------epoch:", epoch, "---------")
        ave_loss = train(train_loader)
        # test(test_loader)
        print("---------AVE LOSS:", ave_loss, "---------")
        re.write("---------EPOCH: {epoch}---AVE LOSS: {ave_loss}----------\n".format(epoch=epoch,ave_loss=ave_loss))
        lst_ave_loss = ave_loss
        if ave_loss > lst_ave_loss:
            count += 1
        if count == 6:
            break
        # with open('test_score.txt', 'r') as f:
        #     evaluation.evaluate(f)



    test(test_loader)
    with open('test_score.txt', 'r') as f:
        evaluation.evaluate(f)

    # len(user_id): 26
    # user_id = ['1350947', '2093983', '16953212', '5947777', '14685063', '996273',
    #            '3390690', '1592695', '4194193', '128163', '10288760', '1484683', '2026230',
    #            '10866560', '2100', '6288704', '1626027', '326261', '3578483', '4014766',
    #            '6686850', '4703374', '366999', '593169', '2363200', '1106763']
    # # len(set(self.user_tmp)): 18
    # # set(self.user_tmp): {'10866560', '16953212', '5947777', '14685063', '996273',
    # #  '4014766', '6686850', '4703374', '366999', '6288704', '593169', '2363200',
    # #  '1592695', '1484683', '128163', '2026230', '1106763', '1626027'}
    # # set(user_id) - user_tmp : {'4194193', '3390690', '3578483', '2093983', '326261', '1350947', '2100', '10288760'}
    # # ['10288760.txt', '14685063.txt', '6686850.txt', '4194193.txt', '996273.txt', '3578483.txt', '2026230.txt', '2100.txt']
    #
    # data.reload_user_history(user_id, '2006-05-01 00:00:00')

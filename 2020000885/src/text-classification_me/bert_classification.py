# -*- encoding: utf-8 -*-
import os
import sys
import pickle
import pandas as pd
import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor
import torch
from sklearn.preprocessing import LabelEncoder
from torch.optim import optimizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from tqdm import tqdm_notebook, trange
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from sklearn.metrics import precision_recall_curve,classification_report
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split


# 早停法
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model, 'checkpoint.pt')
        self.val_loss_min = val_loss


# 标签平滑
class LabelSmoothing(nn.Module):
    def __init__(self, size, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        # self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing  # if i=y的公式
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        """
        x表示输入 (N，M)N个样本，M表示总类数，每一个类的概率log P
        target表示label（M，）
        """
        assert x.size(1) == self.size
        true_dist = x.data.clone()  # 先深复制过来
        # print true_dist
        true_dist.fill_(self.smoothing / (self.size - 1))  # otherwise的公式
        # print true_dist
        # 变成one-hot编码，1表示按列填充，
        # target.data.unsqueeze(1)表示索引,confidence表示填充的数字
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        self.true_dist = true_dist

        return self.criterion(x, Variable(true_dist, requires_grad=False))


# 数据处理类
class DataPrecessForSingleSentence(object):
    """
    对文本进行处理
    """

    def __init__(self, bert_tokenizer, max_workers=10):
        """
        bert_tokenizer :分词器
        dataset        :包含列名为'text'与'label'的pandas dataframe
        """
        self.bert_tokenizer = bert_tokenizer
        # 创建多线程池
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        # 获取文本与标签

    def get_input(self, dataset, max_seq_len=512):
        """
        通过多线程（因为notebook中多进程使用存在一些问题）的方式对输入文本进行分词、ID化、截断、填充等流程得到最终的可用于模型输入的序列。

        入参:
            dataset     : pandas的dataframe格式，包含两列，第一列为文本，第二列为标签。标签取值为{0,1}，其中0表示负样本，1代表正样本。
            max_seq_len : 目标序列长度，该值需要预先对文本长度进行分别得到，可以设置为小于等于512（BERT的最长文本序列长度为512）的整数。

        出参:
            seq         : 在入参seq的头尾分别拼接了'CLS'与'SEP'符号，如果长度仍小于max_seq_len，则使用0在尾部进行了填充。
            seq_mask    : 只包含0、1且长度等于seq的序列，用于表征seq中的符号是否是有意义的，如果seq序列对应位上为填充符号，
                          那么取值为1，否则为0。
            seq_segment : shape等于seq，因为是单句，所以取值都为0。
            labels      : 标签取值为{0,1}，其中0表示负样本，1代表正样本。
        """
        sentences = dataset.iloc[:, 0].tolist()
        labels = dataset.iloc[:, 1].tolist()
        # 切词
        tokens_seq = list(
            self.pool.map(self.bert_tokenizer.tokenize, sentences))
        # 获取定长序列及其mask，多线程处理
        result = list(
            self.pool.map(self.trunate_and_pad, tokens_seq,
                          [max_seq_len] * len(tokens_seq)))
        seqs = [i[0] for i in result]
        seq_masks = [i[1] for i in result]
        seq_segments = [i[2] for i in result]
        return seqs, seq_masks, seq_segments, labels

    def trunate_and_pad(self, seq, max_seq_len, head_ignore=10):
        """
        1. 因为本类处理的是单句序列，按照BERT中的序列处理方式，需要在输入序列头尾分别拼接特殊字符'CLS'与'SEP'，
           因此不包含两个特殊字符的序列长度应该小于等于max_seq_len-2，如果序列长度大于该值需要那么进行截断。
        2. 对输入的序列 最终形成['CLS',seq,'SEP']的序列，该序列的长度如果小于max_seq_len，那么使用0进行填充。

        入参:
            seq         : 输入序列，在本处其为单个句子。
            max_seq_len : 拼接'CLS'与'SEP'这两个特殊字符后的序列长度
            head_ignore : 忽略开头的一些字词

        出参:
            seq         : 在入参seq的头尾分别拼接了'CLS'与'SEP'符号，如果长度仍小于max_seq_len，则使用0在尾部进行了填充。
            seq_mask    : 只包含0、1且长度等于seq的序列，用于表征seq中的符号是否是有意义的，如果seq序列对应位上为填充符号，
                          那么取值为1，否则为0。
            seq_segment : shape等于seq，因为是单句，所以取值都为0。

        """
        # 对超长序列进行截断
        if len(seq) > (max_seq_len - 2):
            seq = seq[0:(max_seq_len - 2)]
        # 分别在首尾拼接特殊符号
        seq = ['[CLS]'] + seq + ['[SEP]']
        # ID化
        seq = self.bert_tokenizer.convert_tokens_to_ids(seq)
        # 根据max_seq_len与seq的长度产生填充序列
        padding = [0] * (max_seq_len - len(seq))
        # 创建seq_mask
        seq_mask = [1] * len(seq) + padding
        # 创建seq_segment
        seq_segment = [0] * len(seq) + padding
        # 对seq拼接填充序列
        seq += padding
        assert len(seq) == max_seq_len
        assert len(seq_mask) == max_seq_len
        assert len(seq_segment) == max_seq_len
        return seq, seq_mask, seq_segment


def pre_process(train_path='../cnews/cnews.train.txt',
                bert_pretrained_path='../chinese_wwm_ext_pytorch',
                header: int=None,
                label_text_columns=['label', 'text'],
                delete_temp=True,
                pretrain_model=None,
                max_seq_len=512
                ):
    """
    header: 第几行为头
    label_text_columns：必须为'label', 'text'对应的列标签
    delete_temp: 是否删除数据缓存。新数据一定要删除缓存
    """
    # 读取数据
    if header is not None:
        train = pd.read_table(train_path, encoding='utf-8',
                              header=header, sep=',')
        train = train[label_text_columns]
    else:
        # todo 这里需要修改，不能 label_text_columns，可能列数不匹配
        train = pd.read_table(train_path, encoding='utf-8',
                              header=header, names=label_text_columns)
        train = train[label_text_columns]

    train.columns = ['label', 'text']
    # 编码标签
    le = LabelEncoder()
    le.fit(train.label.tolist())
    train['label_id'] = le.transform(train.label.tolist())

    # 保存码表
    labeldata = train.groupby(['label', 'label_id']).count().reset_index()
    num_labels = labeldata.shape[0]
    labeldata.to_excel('./train_labels.xlsx', index=None)

    # 将训练数据集拆分为训练集和验证集。
    train_data = train[['text', 'label_id']]
    train, valid = train_test_split(train_data, train_size=SPLIT_RATIO, random_state=SEED)
    train_labels = train.groupby(['label_id']).count().reset_index()
    valid_labels = valid.groupby(['label_id']).count().reset_index()

    # 分词工具
    bert_tokenizer = BertTokenizer.from_pretrained(bert_pretrained_path, do_lower_case=False)
    # 类初始化
    processor = DataPrecessForSingleSentence(bert_tokenizer=bert_tokenizer)

    # 产生训练集输入数据
    if os.path.exists('./train_input.json') and delete_temp is False:
        with open('train_input.json', mode='rt', encoding='UTF-8') as f:
            l = json.loads(f.read())
            seqs, seq_masks, seq_segments, labels = l['seqs'], l['seq_masks'], l['seq_segments'], l['labels']

    else:
        seqs, seq_masks, seq_segments, labels = processor.get_input(
            dataset=train, max_seq_len=max_seq_len)
        train_input_save = {'seqs': seqs, 'seq_masks': seq_masks, 'seq_segments': seq_segments, 'labels': labels}
        with open('train_input.json', mode='wt', encoding='UTF-8') as f1:
            json.dump(train_input_save, f1)

    # 转换为torch tensor
    t_seqs = torch.tensor(seqs, dtype=torch.long)
    t_seq_masks = torch.tensor(seq_masks, dtype = torch.long)
    t_seq_segments = torch.tensor(seq_segments, dtype = torch.long)
    t_labels = torch.tensor(labels, dtype=torch.long)

    train_data = TensorDataset(t_seqs, t_seq_masks, t_seq_segments, t_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloder = DataLoader(dataset=train_data, sampler=train_sampler,batch_size = BATCH_SIZE)


    # 产生验证集输入数据
    if os.path.exists('./valid_input.json') and delete_temp is False:
        with open('valid_input.json', mode='rt', encoding='UTF-8') as f:
            l = json.loads(f.read())
            seqs, seq_masks, seq_segments, labels = l['seqs'], l['seq_masks'], l['seq_segments'], l['labels']

    else:
        seqs, seq_masks, seq_segments, labels = processor.get_input(
            dataset=valid, max_seq_len=max_seq_len)
        valid_input_save = {'seqs': seqs, 'seq_masks': seq_masks, 'seq_segments': seq_segments, 'labels': labels}
        with open('valid_input.json', mode='wt', encoding='UTF-8') as f1:
            json.dump(valid_input_save, f1)
    # 转换为torch tensor
    t_seqs = torch.tensor(seqs, dtype=torch.long)
    t_seq_masks = torch.tensor(seq_masks, dtype = torch.long)
    t_seq_segments = torch.tensor(seq_segments, dtype = torch.long)
    t_labels = torch.tensor(labels, dtype = torch.long)

    valid_data = TensorDataset(t_seqs, t_seq_masks, t_seq_segments, t_labels)
    valid_sampler = RandomSampler(valid_data)
    valid_dataloder = DataLoader(dataset= valid_data, sampler= valid_sampler,batch_size = BATCH_SIZE)

    if pretrain_model is not None:
        model = pretrain_model
    else:
        # 加载预训练的bert模型
        model = BertForSequenceClassification.from_pretrained(bert_pretrained_path, num_labels=num_labels)

    if GPUID == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:' + str(GPUID))  # gpu版本
    model = model.to(device)
    # 待优化的参数
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params':[p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay':0.01},
        {'params':[p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay':0.0}
    ]

    steps = len(train_dataloder) * EPOCHS
    optimizer = BertAdam(optimizer_grouped_parameters, lr=LR, warmup= 0.1 , t_total= steps)
    loss_function = LabelSmoothing(num_labels, 0.1)

    return model, train_dataloder, valid_dataloder, loss_function, optimizer,\
           device, num_labels


def train(model, train_dataloder, valid_dataloder, loss_function, optimizer,\
           device, num_labels,
                save_path='./job_fine_tuned_bert.pth'):
    # 存储loss
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    patience = 20
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # 模型训练
    for i in trange(EPOCHS, desc='Epoch'):

        model.train()  # 训练
        for step, batch_data in enumerate(train_dataloder):
            batch_data = tuple(t.to(device) for t in batch_data)
            batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels = batch_data
            logits = model(
                batch_seqs, batch_seq_masks, batch_seq_segments, labels=None)
            logits = torch.nn.functional.log_softmax(logits, dim=1)
            # loss_function = CrossEntropyLoss()
            loss = loss_function(logits, batch_labels)
            loss.backward()
            train_losses.append(loss.item())
            print("\r step: %d / %d, loss: %f" % (step, len(train_dataloder), loss), end='')
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

        model.eval()  # 验证
        for step, batch_data in enumerate(valid_dataloder):
            with torch.no_grad():
                batch_data = tuple(t.to(device) for t in batch_data)
                batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels = batch_data

                logits = model(
                    batch_seqs, batch_seq_masks, batch_seq_segments, labels=None)
                logits = torch.nn.functional.log_softmax(logits, dim=1)
                # loss_function = CrossEntropyLoss()
                loss = loss_function(logits, batch_labels)
                valid_losses.append(loss.item())
        torch.cuda.empty_cache()
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        if step % 20 == 0:
            print("train_loss:%f, valid_loss:%f" % (train_loss, valid_loss))

        # 重置训练损失和验证损失
        train_losses = []
        valid_losses = []

        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early Stopping")
            break

    torch.save(model, open(save_path, "wb"))

    # 绘制 loss 图
    fig = plt.figure(figsize=(8,6))
    plt.plot(range(1, len(avg_train_losses)+1), avg_train_losses, label='Training Loss')
    plt.plot(range(1, len(avg_valid_losses)+1), avg_valid_losses, label='Validation Loss')
    # find the position of lowest validation loss
    minposs = avg_valid_losses.index(min(avg_valid_losses))+1
    # plt.axvline(minposs, linestyle='--', color = 'r', lable='Early Stopping Checkpoint')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig('loss_plot.png', bbox_inches='tight')

    return model


if __name__ == '__main__':
    # ------------------------------------
    # 超参数
    SPLIT_RATIO = 0.9  # 训练和验证集的比例
    MAX_SEQ_LEN = 512
    LR = 1e-6
    BATCH_SIZE = 16
    SEED = 0
    EPOCHS = 100
    GPUID = 1

    # pretrain_model = torch.load('./2-2_seq256_fine_tuned_bert.pth')
    pretrain_model = None
    save_path = './2-4-'+str(MAX_SEQ_LEN)+'_supplement_fine_tuned_bert.pth'
    model, train_dataloder, valid_dataloder, loss_function, optimizer, \
        device, num_labels = pre_process(train_path='../labeled_data_supplement.csv',
                                         header=0,
                                         label_text_columns=['class_label', 'content'],
                                         delete_temp=False,
                                         pretrain_model=pretrain_model,
                                         max_seq_len=MAX_SEQ_LEN)
    model, _, _ = train(model, train_dataloder, valid_dataloder, loss_function, optimizer, \
          device, num_labels, save_path=save_path)
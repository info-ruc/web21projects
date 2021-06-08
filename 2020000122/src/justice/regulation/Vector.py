#!/usr/bin/python3
# -*- coding:utf-8 -*-
import nltk
import matchzoo as mz
import pandas as pd
import re  # 这个可以不用
from sklearn import preprocessing  # 用于正则化
import numpy as np 
preprocessor = mz.preprocessors.BasicPreprocessor()  # 定义一个数据处理器，有四种处理器，Basic是通用的、基础的数据处理器，可看官方文档，这里不做解说


def data_convert():
    data = []
    data_type = 'test'
    with open('msr_paraphrase_%s.txt' % data_type, 'r', encoding='utf-8')as f:
        for line in f.readlines()[1:]:  # 这个是为了忽略标题
            line = line.strip().split('\t')
            data.append([line[1], line[3], line[2], line[4], line[0]])  # 是为了方便matchzoo的输入格式
    data = pd.DataFrame(data)
    train_data_path = 'my_test_data.csv'
    data.to_csv(train_data_path, header=False, index=False, sep='\t')

def load_data(data_path):
	df_data = pd.read_csv(data_path, sep='\t', header=None)
	df_data = pd.DataFrame(df_data.values, columns=['id_left', 'text_left', 'id_right', 'text_right', 'label'])
	df_data = mz.pack(df_data)
	return df_data


def build():
    model = mz.models.DUET()  # 同样，DUET网络可看官网的论文，这里不做解释；同样，模型的参数不做解释，官方文档有
    ranking_task = mz.tasks.Ranking(loss=mz.losses.RankCrossEntropyLoss(num_neg=1))  # 定义损失函数，这里采用的是排序交叉熵损失函数，它还有一个分类交叉熵损失函数，看你如何定义你的数据
    model.params['input_shapes'] = preprocessor.context['input_shapes']
    model.params['embedding_input_dim'] = preprocessor.context['vocab_size']  # 如果版本较老，这里需要加1，因为要考虑一个UNK的字符，如果版本较新，这个以更新解决
    model.params['embedding_output_dim'] = 300 
    model.params['task'] = ranking_task
    model.params['optimizer'] = 'adam'
    model.params['padding'] = 'same'
    model.params['lm_filters'] = 32
    model.params['lm_hidden_sizes'] = [32]
    model.params['dm_filters'] = 32
    model.params['dm_kernel_size'] = 3
    model.params['dm_d_mpool'] = 3
    model.params['dm_hidden_sizes'] = [32]
    model.params['activation_func'] = 'relu'
    model.params['dropout_rate'] = 0.32
    model.params['embedding_trainable'] = True
    model.guess_and_fill_missing_params(verbose=0)
    model.params.completed()
    model.build()
    model.backend.summary()
    model.compile()
    return model




if __name__ == '__main__':
    nltk.download('punkt')
    train_data_path = 'my_train_data.csv'
    test_data_path = 'my_test_data.csv'
    train_data = load_data(train_data_path)  # 这里就是上面数据格式转换的训练集和测试集路径
    test_data = load_data(test_data_path)
    
    train_dev_split = int(len(train_data) * 0.9)  # 验证集占训练数据的0.1
    train = train_data[:train_dev_split]
    dev = train_data[train_dev_split:]
    train_pack_processed = preprocessor.fit_transform(train)  # 其实就是做了一个字符转id操作，所以对于中文文本，不需要分词
    dev_pack_processed = preprocessor.transform(dev)  
    test_pack_processed = preprocessor.transform(test_data)
    train_data_generator = mz.DataGenerator(train_pack_processed, batch_size=32, shuffle=True)  # 训练数据生成器

    test_x, test_y = test_pack_processed.unpack()
    dev_x, dev_y = dev_pack_processed.unpack()


    model = build()
    batch_size = 32

    evaluate = mz.callbacks.EvaluateAllMetrics(model, x=dev_x, y=dev_y, batch_size=batch_size)
    model.fit_generator(train_data_generator, epochs=5, callbacks=[evaluate], workers=5, use_multiprocessing=False)
    y_pred = model.predict(test_x)

    left_id = test_x['id_left']
    right_id = test_x['id_right']
    assert (len(left_id) == len(left_id))
    assert (len(left_id) == len(y_pred))
    assert (len(test_y) == len(y_pred))
    Scale = preprocessing.MinMaxScaler(feature_range=(0, 1))  # 对结果做规范化
    y_pred = Scale.fit_transform(y_pred)


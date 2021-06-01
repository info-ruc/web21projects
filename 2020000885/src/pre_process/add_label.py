# -*- encoding: utf-8 -*-
import numpy as np
import pandas as pd
import jieba
import jieba.analyse
import json
from tqdm import tqdm


def get_freqent_word_dict(labeled_data_path='../labeled_data.csv'):
    """

    """
    # 读取数据
    labeled_data = pd.read_csv(labeled_data_path, encoding='utf-8')
    labels = set(labeled_data['class_label'].to_list())
    freq_voca_count = {}  # 每个label中重要词统计
    for each in labels:
        freq_voca_count[each] = {}

    for i in tqdm(range(labeled_data.shape[0])):
        label = labeled_data['class_label'][i]
        top_words = jieba.analyse.extract_tags(labeled_data['content'][i], allowPOS='n')
        for each in top_words:
            if each in freq_voca_count[label]:
                freq_voca_count[label][each] += 1
            else:
                freq_voca_count[label][each] = 1
    # 取前50个最常见的词
    for each in freq_voca_count:
        print(each)
        freq_voca_count[each] = sorted(freq_voca_count[each].items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:50]

    return freq_voca_count


def add_label_through_weight():
    """
    通过 TF-IDF 方法判断权重
    """
    stored_path = './processed_data.csv'
    processed_labels = {'家居', '房产', '教育', '时尚', '时政', '科技', '财经'}  # 已经处理的类别
    key_labels = {'游戏', '娱乐', '体育'}  # 目标类别
    processed_data = pd.DataFrame(columns=['class_label', 'content'])  # 处理后的data
    unlabeled_data = pd.read_csv('../unlabeled_data.csv')
    for i in tqdm(range(unlabeled_data.shape[0])):
        top_words = dict(jieba.analyse.extract_tags(unlabeled_data['content'][i], topK=30,
                                               allowPOS='n', withWeight=True))
        if i > 1000:
            break
        # 如果存在processed_labels中词，则转到下一个
        if processed_labels & top_words.keys():
            continue
        # top_words和key_labels 取交集，如果有多个元素，判断哪个权重大
        overlap = list(set(top_words.keys()) & set(key_labels))
        if overlap:
            overlap_weights = [top_words[label_] for label_ in overlap]
            this_label = overlap[overlap_weights.index(max(overlap_weights))]
            processed_data = processed_data.append([{'class_label': this_label,
                                                     'content': unlabeled_data['content'][i]}], ignore_index=True)

    processed_data.to_csv(stored_path)


def add_label_through_weight_1():
    """
    通过 人工设定优先级 方法判断权重
    """
    stored_path = './processed_data.csv'
    processed_labels = {'家居', '房产', '教育', '时尚', '时政', '科技', '财经'}  # 已经处理的类别
    key_labels = ['体育', '娱乐', '游戏']  # 目标类别优先排序
    processed_data = pd.DataFrame(columns=['class_label', 'content'])  # 处理后的data
    unlabeled_data = pd.read_csv('../unlabeled_data.csv')
    for i in tqdm(range(unlabeled_data.shape[0])):
        top_words = dict(jieba.analyse.extract_tags(unlabeled_data['content'][i], topK=30,
                                               allowPOS='n', withWeight=True))

        # 如果存在processed_labels中词，则转到下一个
        if processed_labels & top_words.keys():
            continue
        # top_words和key_labels 取交集，如果有多个元素，判断哪个权重大
        overlap = list(set(key_labels) & set(top_words.keys()))
        for each in key_labels:

            if each in overlap:
                processed_data = processed_data.append([{'class_label': each,
                                                     'content': unlabeled_data['content'][i]}], ignore_index=True)
                break

    processed_data.to_csv(stored_path)


def add_label_simple(stored_path='./processed_data.csv',
                     delete_processed_labels=False):
    """
    通过 判断存在即打标签
    """

    processed_labels = {'家居', '房产', '教育', '时尚', '时政', '科技', '财经'}  # 已经处理的类别
    stop_words = {'股市', '出台', '本科志愿', '电脑病', '消费者', '景点'}  # 含有这些词就肯定不属于下面三类
    key_labels = {
        '娱乐': ['娱乐', '电影', '影片', '主演', '上映', '票房', '剧场'],
        '体育': ['体育', '篮板', '球员', '球队'],
        '游戏': ['游戏', '玩家']}  # 目标类别优先排序
    # key_labels = {
    #     '娱乐': ['娱乐'],
    #     '体育': ['体育'],
    #     '游戏': ['游戏']}  # 目标类别优先排序
    processed_data = pd.DataFrame(columns=['class_label', 'content'])  # 处理后的data
    unlabeled_data = pd.read_csv('../unlabeled_data.csv')
    for i in tqdm(range(unlabeled_data.shape[0])):
        # if i > 500:
        #     break
        if delete_processed_labels:
            # 如果存在processed_labels中词，则转到下一个
            flag = False
            for each in processed_labels | stop_words:
                if each in unlabeled_data['content'][i]:
                    flag = True
            if flag:
                continue
        # 如果存在相关词，则判断属于目标类
        for each_key in key_labels.keys():
            flag = False
            for each_word in key_labels[each_key]:
                if each_word in unlabeled_data['content'][i]:
                    processed_data = processed_data.append([{'class_label': each_key,
                                                             'content': unlabeled_data['content'][i]}], ignore_index=True)
                    flag = True
                    break
            if flag:
                break

    processed_data.to_csv(stored_path)
    return processed_data



processed_data = add_label_simple(delete_processed_labels=True)
# freq_voca_count = get_freqent_word_dict('./processed_data.csv')
# with open('frequent_words.json', mode='wt', encoding='UTF-8') as f1:
#     json.dump(freq_voca_count, f1, ensure_ascii=False)

# todo 对unlabel_data, 使用预训练模型把目标类别提取，然后提取排行靠前的关键词，作为判定依据再次进行筛选





# encouding=utf-8

import jieba
from tqdm import tqdm

result_path = "train_data_cut.txt"
file_path = "train_data.txt"
sent_count = 0

with open(file_path, 'r', encoding='UTF-8') as read_f:
    for line in read_f:
        sent_count += 1

with tqdm(total=sent_count) as pbar:
    with open(result_path, 'w+', encoding='UTF-8') as result_f:
        with open(file_path, 'r', encoding='UTF-8') as read_f:
            for line in read_f:
                seg_list = jieba.cut(line, cut_all=False)
                result_f.write(" ".join(seg_list))
                pbar.update(1)


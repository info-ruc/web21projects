import os
import json

def load_vocabs(data_path):
    dict_path = os.path.join(data_path, 'dict.json')
    word2id = json.load(open(dict_path, 'r', encoding='utf-8'))
    id2word = {word2id[key]:key for key in word2id}
    print('Vocabulary Size', len(word2id))
    return word2id, id2word
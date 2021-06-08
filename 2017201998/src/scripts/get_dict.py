import os
import json
import copy
import nltk
from nltk.corpus import wordnet

def get_wordnet_pos(tag):
    if tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    else:
        return None

def get_words(sentense, flag):
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']
    tokens_ = []
    tokens = nltk.word_tokenize(sentense)
    s = 0
    #punc = ['.',',','%','(',')']
    for token in tokens:
        if token == '':
            continue
        if '.' in token or ',' in token or '%' in token or '(' in token or ')' in token or ';' in token:
            continue
        if '/' in token:
            ts = token.split('/')
            for t in ts:
                if t != '':
                    tokens_.append(t)
        else:
            tokens_.append(token)
    tokens = copy.deepcopy(tokens_)
    tokens_ = []
    for token in tokens:
        if '-' in token:
            ts = token.split('-')
            for t in ts:
                if t != '':
                    tokens_.append(t)
            
        else:
            tokens_.append(token)
    #print(tokens_)
    tagged_tokens = nltk.pos_tag(tokens_)
    wnl = nltk.stem.WordNetLemmatizer()
    words = []
    for tag in tagged_tokens:
        wordnet_pos = get_wordnet_pos(tag[1])
        if wordnet_pos != None:
            word = wnl.lemmatize(tag[0], pos=wordnet_pos) # verb/noun Lemmatization
        else:
            word = tag[0]
        words.append(word)
    if flag == 'title':
        return words
    elif flag == 'abs':
        return [word for word in words if word not in stop_words]
        
def filtering(data,dic):
    dic_ = {}
    data_ = {}
    dic_['<BOS>'] = 1
    dic_['<EOS>'] = 2
    i = 3
    for k,v in dic.items():
        if v > 10:
            dic_[k] = i
            i += 1
    
    for k,v in data.items():
        title = []
        abst = []
        for t in v['title']:
            if t in dic_:
                title.append(t)
        for a in v['abstract']:
            if a in dic_:
                abst.append(a)
        data_[k] = {'title': title, 'abstract': abst}
    return data_,dic_

def split_dataset(datas):
    print('Total data ',len(datas))
    train = dict(list(datas.items())[:int(len(datas)/10*9)])
    d = json.dumps(train)
    f = open('train.json', 'w') 
    f.write(d)
    f.close()

    val_1 = dict(list(datas.items())[int(len(datas)/10*9):int(len(datas)/10*9.5)])
    d = json.dumps(val_1)
    f = open('val.json', 'w') 
    f.write(d)
    f.close()

    val_2 = dict(list(datas.items())[int(len(datas)/10*9.5):])
    d = json.dumps(val_2)
    f = open('test.json', 'w') 
    f.write(d)
    f.close()
    return

if __name__ == '__main__':
    f = open('data/ta_data.json', 'r', encoding='utf-8')
    data = json.load(f)
    data_ = {}
    dic = {}
    i = 1

    wnl = nltk.stem.WordNetLemmatizer()
    for k,v in data.items():
        t = get_words(v['title'].lower(),'title')
        a = get_words(v['abstract'].lower(),'abs')
        for word in t:
            if word not in dic:
                dic[word] = 1
            else:
                dic[word] += 1
        for word in a:
            if word not in dic:
                dic[word] = 1
            else:
                dic[word] += 1

        data_[k] = {'title': t, 'abstract': a}

    data_,dic = filtering(data_,dic)

    split_dataset(data_)
    
    d = json.dumps(dic)
    f = open('dict.json', 'w') 
    f.write(d)
    f.close()
    print(len(dic))
    '''
    f = open('train.json', 'r', encoding='utf-8')
    data = json.load(f)
    print(len(data))
    f = open('val.json', 'r', encoding='utf-8')
    data = json.load(f)
    print(len(data))
    f = open('test.json', 'r', encoding='utf-8')
    data = json.load(f)
    print(len(data))
    '''
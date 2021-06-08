import json
import torch
import torch.utils.data

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, vocabs, rev_vocabs, g_max_len, input_max_len, set_type):
        # set parameter
        self.g_max_len = g_max_len
        self.input_max_len = input_max_len
        self.vocabs = vocabs
        self.rev_vocabs = rev_vocabs
        self.set_type = set_type

        self.BOS = vocabs['<BOS>']
        self.EOS = vocabs['<EOS>']

        # load data
        self.load_data(data_path)


    def load_data(self, data_path):
        self.datas = []
        dropdata = 0
        data = json.load(open(data_path, 'r', encoding='utf-8'))
        for k,v in data.items():
            if len(v["title"])!= 0 and len(v["abstract"])!= 0:
                self.datas.append({'id':k, 'title':v["title"], 'abstract':v["abstract"]})
            else:
                dropdata += 1
        print('Total datas ', len(self.datas), 'drop ', dropdata, ' data')

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        # training and validation
        data = self.datas[index]
        abstract, a_len = self.padding(data['abstract'], self.input_max_len)
        title, t_len = self.padding(data['title'], self.g_max_len)
        return abstract, title

    def get_data(self, index):
        # test mode
        data = self.datas[index]
        abstract, a_len = self.padding(data['abstract'], self.input_max_len)
        paperId = data['id']
        return paperId, abstract, self.datas[index]['title']

    def padding(self, sent, max_len):
        if len(sent) > max_len-3:
            sent = sent[:max_len-3]
        text = list(map(lambda t: self.vocabs.get(t), sent))
        text = [self.BOS] + text + [self.EOS]
        length = len(text)
        T = torch.cat([torch.LongTensor(text), torch.zeros(max_len - length).long()])
        return T, length

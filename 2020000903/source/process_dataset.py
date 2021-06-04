import pickle
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--data_path",type=str,required=True)
parser.add_argument('--bin_path',type=str,required=True)
args=parser.parse_args()
file_path=args.data_path

bin_path=args.bin_path
data_name=['test','valid']
languages=['en','de']
for language in languages:
    current_dict={}
    current_dict['<padding>']=0
    current_dict['<sos>']=1
    current_dict['<eos>']=2
    current_dict['<unk>']=3
    current_file=file_path+'train.'+language
    lines=open(current_file,encoding='utf-8').readlines()
    indexs=[]
    for line in lines:
        line=line.strip().split(' ')
        sentence_index=[]
        for word in line:
            if word not in current_dict.keys():
                current_dict[word]=len(current_dict.keys())
            sentence_index.append(current_dict[word])
        indexs.append(sentence_index)
    pickle.dump(indexs,open(bin_path+'train.'+language,'wb'))
    with open(bin_path+'train.'+language+'word','w') as f:
        for key in indexs:
            f.write(' '.join([str(each) for each in key])+'\n')
    with open(bin_path+'dict'+language+'word','w') as f:
        for word in current_dict.keys():
            f.write(word+'\n')
    pickle.dump(current_dict,open(bin_path+'dict.'+language,'wb'))
    for file in data_name:
        lines=open(file_path+file+'.'+language,encoding='utf-8').readlines()
        indexes=[]
        for line in lines:
            sentence_index=[]
            line=line.strip().split(' ')
            for word in line:
                if word not in current_dict.keys():
                    sentence_index.append(3)
                else:
                    sentence_index.append(current_dict[word])
            indexes.append(sentence_index)
        with open(bin_path + file+'.' + language + 'word', 'w') as f:
            for key in indexs:
                f.write(' '.join([str(each) for each in key]) + '\n')
        pickle.dump(indexes,open(bin_path+file+'.'+language,'wb'))
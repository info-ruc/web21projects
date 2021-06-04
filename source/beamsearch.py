from Model import Transformer
import torch
import math
import heapq
import pickle
from dataload import Dataload
from torch.nn.utils.rnn import pad_sequence

from nltk.translate.bleu_score import corpus_bleu
import argparse
device="cuda:0"
# def greedy_next_words(total_sentence: list, is_ends: list, predict: torch.Tensor, sentence_length: list, prob: list,
#                    beam_size: int,is_first=False):

def get_next_words(total_sentence: list, is_ends: list, predict: torch.Tensor, sentence_length: list, prob: list,
                   beam_size: int,is_first=False):
    """
    used to get max k words in predict
    :param predict:batchsize*beam_size,seq_len,embedding
    :param beam_size:int
    :return:
    """
    next_predict = []
    next_score = []
    next_is_end = []
    next_length = []
    tmp_length = []
    tmp_predict = []
    tmp_score = []
    assert len(total_sentence) == predict.shape[0]
    if is_first:
        current_batch=1
    else:
        current_batch=beam_size
    tmp_is_end = []
    for i in range(predict.shape[0]):
        current_predict = predict[i][-1]
        scores, indexs = torch.topk(torch.log_softmax(current_predict, -1), beam_size, -1,largest=True)
        # scores=scores[0].detach().cpu().numpy()
        # indexs=indexs[0].detach().cpu().numpy()
        if total_sentence[i][-1] == 2 or is_ends[i]:  # Does this sentence have reached its end
            tmp_predict.append(total_sentence[i] + [0])
            tmp_score.append(prob[i])
            tmp_length.append(sentence_length[i])
            tmp_is_end.append(True)
        else:
            for j in range(beam_size):
                tmp_predict.append(total_sentence[i] + [indexs[j].cpu().item()])
                score = (prob[i] * math.pow(sentence_length[i], 0.7) + scores[j]) / math.pow(sentence_length[i] + 1, 0.7)
                # score=prob[i]+scores[j]
                tmp_length.append(sentence_length[i] + 1)
                tmp_score.append(score)
                tmp_is_end.append(False)

        if (i + 1) % current_batch == 0:
            max_score = heapq.nlargest(beam_size, tmp_score)
            indexs = [tmp_score.index(i) for i in max_score]
            for index in indexs:
                next_predict.append(tmp_predict[index])
                if next_predict[-1][-1] == 2 or tmp_is_end[index]:
                    next_is_end.append(True)
                else:
                    next_is_end.append(False)
                next_length.append(tmp_length[index])
                next_score.append(tmp_score[index])
            tmp_score = []
            tmp_length = []
            tmp_predict = []
            tmp_is_end = []
    # assert len(next_predict) == len(total_sentence)
    assert len(next_predict) == len(next_score)
    return next_predict, next_score, next_is_end, next_length


def beamsearch(model, sentence: torch.Tensor, beam_size, seq_len, device):
    """

    :param sentence:
    :param beam_size:
    :param seq_len:
    :param device:
    :return:
    """
    # sentence = sentence.to(device)
    btz = sentence.shape[0]
    orig_predict = [[1]] * btz
    total_score = [0] * btz
    seq_length = [0] * btz
    is_end = [False] * btz
    sentences = []
    for i in range(sentence.shape[0]):
        for j in range(beam_size):
            sentences.append(sentence[i])
    sentences = torch.stack(sentences,0).to(device)  # shape:bathc*beam_size,seq_len
    encode_masks = (sentences == 0).to(device).view(btz*beam_size,1,1,sentence.shape[-1])
    encode_mask=(sentence==0).to(device).view(btz,1,1,sentence.shape[-1])
    predict_words = [[]]
    sentence=sentence.to(device)
    memory = model.Encoder(sentences, encode_masks)
    tmp_memory=model.Encoder(sentence,encode_mask)
    for i in range(seq_len):
        if i ==0:
            current_memory=tmp_memory
            current_mask=encode_mask
            is_first=True
        else:
            current_memory=memory
            current_mask=encode_masks
            is_first=False
        input_tensor = torch.tensor(orig_predict).to(device)  # shape [batch_size,1]
        ahead_mask=torch.triu(torch.ones(i+1,i+1),1).bool().to(device)
        predict = model.Decoder(input_tensor, current_memory, current_mask,ahead_mask=ahead_mask)  # shape [batch_size,1]
        orig_predict, total_score, is_end, seq_length = get_next_words(orig_predict, is_end, predict, seq_length,
                                                                       total_score, beam_size,is_first)
        if all(is_end):
            break
    current_score = []
    return_list = []
    for i in range(len(orig_predict)):
        current_score.append(total_score[i])
        if (i + 1) % beam_size == 0:
            max_score = max(current_score)
            index = current_score.index(max_score)
            return_list.append(orig_predict[i - beam_size+1 + index])
            current_score = []
    return return_list
def collate_fn(batch):
    srcs=[]
    trgs=[]
    for s,t in batch:
        srcs.append(s)
        trgs.append(t.numpy())
    srcs=pad_sequence(srcs,batch_first=True,padding_value=0)
    return srcs,trgs
def add_(path):
    if path[-1]!='/':
        path+='/'
    return path
from torch.utils.data import DataLoader
from tqdm import tqdm
parser=argparse.ArgumentParser()
parser.add_argument('--bin_path',type=str,required=True)
parser.add_argument('--model_path',type=str,required=True)
parser.add_argument('--src_lang',type=str,required=True)
parser.add_argument('--tgt_lang',type=str,required=True)
args=parser.parse_args()


src_dict = pickle.load(open(add_(args.bin_path)+'dict.'+args.src_lang,'rb'))
trg_dict=pickle.load(open(add_(args.bin_path)+'dict.'+args.tgt_lang,'rb'))
model = Transformer(src_vocab_size=len(src_dict.keys()),tgt_vocab_size=len(trg_dict.keys()),encoder_layer_num=6,decoder_layer_num=6,hidden_size=512,feedback_size=2048,num_head=8,dropout=0.1,device=device)
model=model.to(device)
model.load_state_dict(torch.load(args.model_path))
dataload=DataLoader(Dataload(add_(args.bin_path)+'test',src=args.src_lang,trg=args.tgt_lang),batch_size=32,collate_fn=collate_fn)
real=[]
predict=[]
pbtr=tqdm(total=len(dataload))
with torch.no_grad():
    model.eval()
    for src,trg in dataload:
        src=src.to(device)
        predicts=beamsearch(model,src,1,100,device=device)

        for i in range(len(predicts)):
            while 0 in predicts[i]:
                predicts[i].remove(0)
        predict=predict+predicts
        real=real+trg
        tmp_bleu=corpus_bleu(list_of_references=[[each] for each in trg],hypotheses=predicts)
        pbtr.set_postfix({'bleu':tmp_bleu})
        print(tmp_bleu)
        pbtr.update(1)
print("Bleu score:{}".format(corpus_bleu(list_of_references=[[each] for each in real],hypotheses=predict)))

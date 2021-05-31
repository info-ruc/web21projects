"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Caption evaluation helper
"""
from time import time

import torch
from horovod import torch as hvd
from tqdm import tqdm

from .logger import LOGGER
from .misc import NoOp
from .distributed import all_gather_list
from pytorch_pretrained_bert import BertTokenizer
from .const import START_TOKEN, END_TOKEN, COCO_MAX_DECODING_LEN, XYB_MAX_DECODING_LEN

import json
import argparse
from builtins import dict
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

class COCOEvalCap:
    """
    COCOEvalCap code is adopted from https://github.com/tylin/coco-caption
    """

    def __init__(self, coco, coco_res):
        self.eval_imgs = []
        self.eval = dict()
        self.img_to_eval = dict()
        self.coco = coco
        self.coco_res = coco_res

    def evaluate(self):
        gts = self.coco
        res = self.coco_res

        # =================================================
        # Set up scorers
        # =================================================
        print("tokenization...")
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)
        # json.dump(res, open('/data2/ruc_vl_pretrain/caption_tmp/aic1000_val_refs_tokenized.json', 'w', encoding='utf-8'), ensure_ascii=False)
        # json.dump(gts, open('/data2/ruc_vl_pretrain/caption_tmp/aic1000_val_gts_tokenized.json', 'w', encoding='utf-8'), ensure_ascii=False)  
        # =================================================
        # Set up scorers
        # =================================================
        print("setting up scorers...")
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            # (Spice(), "SPICE"),
        ]

        # =================================================
        # Compute scores
        # =================================================
        print('computing caption scores')
        for scorer, method in scorers:
            # print("computing %s score..." % (scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.set_eval(sc, m)
                    self.set_img_to_eval_imgs(scs, gts.keys(), m)
                    # print("%s: %0.4f" % (m, sc))
            else:
                self.set_eval(score, method)
                self.set_img_to_eval_imgs(scores, gts.keys(), method)
                # print("%s: %0.4f" % (method, score))
        self.set_eval_imgs()
        #  anwen hu 2020/9/16
        """for img_id in res.keys():
            # print('res_id', res_id)
            hypo = res[img_id]
            gt_captions = gts[img_id]
            cider = self.img_to_eval[img_id]['CIDEr']
            if cider*100 < 20:
                print(img_id, cider, hypo)
                print(gt_captions)
                print('=================')"""

    def set_eval(self, score, method):
        self.eval[method] = score

    def set_img_to_eval_imgs(self, scores, img_ids, method):
        for img_id, score in zip(img_ids, scores):
            if img_id not in self.img_to_eval:
                self.img_to_eval[img_id] = dict()
                self.img_to_eval[img_id]["image_id"] = img_id
            self.img_to_eval[img_id][method] = score

    def set_eval_imgs(self):
        self.eval_imgs = [eval for img_id, eval in self.img_to_eval.items()]


@torch.no_grad()
def caption_eval(gts, refs):
    eval_obj = COCOEvalCap(gts, refs)
    eval_obj.evaluate()
    # print('evaluation done')
    eval_log = eval_obj.eval
    return eval_log


def bert_id2token(tokenizer, ids):
    tokens = tokenizer.convert_ids_to_tokens(ids)
    tokens = list(map(lambda x: '@@'+x if not x.isalpha() else x, tokens))
    return tokens

def reconstruct_sent(tokens):
    sent = []
    i=0
    while i<len(tokens):
        # print(i)
        t = tokens[i].replace('@@', '')
        j = i+1
        while j < len(tokens) and tokens[j].replace('@@', '').startswith('##'):
            # print(j, tokens[j])
            t += tokens[j].replace('@@', '').replace('##', '')
            j+=1
        sent.append(t)
        i=j
    return sent

@torch.no_grad()
def evaluate(model, eval_loader, output_path, opts):
    toker = BertTokenizer.from_pretrained(
        opts.toker, do_lower_case='uncased' in opts.toker)
    st = time()
    LOGGER.info("start running Caption evaluation ...")
    if opts.cap_type is not None and opts.cap_type == 'xyb':
        max_seq_len = XYB_MAX_DECODING_LEN
    else:
        max_seq_len = COCO_MAX_DECODING_LEN
    sep_predicted_matrix = inference(model, eval_loader, max_seq_len, batch_size=opts.eval_batch_size)  # img_num * max_decoding_len
    dset = eval_loader.dataset
    predicted_matrix = hvd.allgather(sep_predicted_matrix)
    # predicted_matrix = sep_predicted_matrix
    all_imgs = [img for imgs in all_gather_list(dset.imgs) for img in imgs]
    # all_imgs = dset.imgs
    gts = {}
    refs = {}
    # predictd_matrix = predicted_matrix
    # print('len(predicted_matrix)', len(predicted_matrix))
    # print('len(all imgs)', len(all_imgs))
    assert len(predicted_matrix) == len(all_imgs)
    gt_cap_num = 0
    for i in range(len(predicted_matrix)):
        img_fname = all_imgs[i]
        id_seq = predicted_matrix[i]
        predicted_id_seq = []
        for j in range(len(id_seq)):
            if id_seq[j] == END_TOKEN:
                break
            else:
                # print(type(id_seq[j])) tensor
                predicted_id_seq.append(int(id_seq[j].cpu()))
        # print(predicted_id_seq)
        predicted_tokens = bert_id2token(toker, predicted_id_seq)
        # print('ref tokens:', predicted_tokens)
        refs[img_fname] = [{'image_id':img_fname,'caption':' '.join(reconstruct_sent(predicted_tokens))}]
        gts[img_fname] = []
        for _id in dset.img2txts[img_fname]:
            gt_cap_num += 1
            gt_example = dset.txt_db[_id]
            if 'toked_caption' in gt_example.keys():
              ground_tokens = gt_example['toked_caption']
            else:
              ground_tokens = gt_example['tokens']
            # print('gt tokens:', ground_tokens)
            gts[img_fname].append({'image_id':img_fname,'caption':' '.join(reconstruct_sent(ground_tokens))})
        # print(refs[img_fname])
        # print(gts[img_fname])
    if output_path is not None:
        json.dump(refs, open(output_path+'_refs.json', 'w', encoding='utf-8'), ensure_ascii=False)
        json.dump(gts, open(output_path+'_gts.json', 'w', encoding='utf-8'), ensure_ascii=False)
    """if hvd.rank()!=0:
       return {}"""
    eval_log = caption_eval(gts, refs)
    tot_time = time()-st
    if hvd.rank() == 0:
        LOGGER.info(f"evaluation {int(len(predicted_matrix))} images, {int(gt_cap_num)} GT captions")
        LOGGER.info(f"evaluation finished in {int(tot_time)} seconds")
    return eval_log


# anwen hu 2020/12/25: evaluate multiple dataset
def evaluate_all(output_pathes):
    all_refs = {}
    all_gts = {}
    img_num = 0
    gt_cap_num = 0
    for output_path in output_pathes:
        refs = json.load(open(output_path + '_refs.json', 'r', encoding='utf-8'))
        gts = json.load(open(output_path + '_gts.json', 'r', encoding='utf-8'))
        for k, v in refs.items():
            assert k not in all_refs.keys()
            all_refs[k] = v
        for k, v in gts.items():
            assert k not in all_gts.values()
            all_gts[k] = v
            img_num += 1
            gt_cap_num += len(v)

    if hvd.rank() != 0:
       return {}
    eval_log = caption_eval(all_gts, all_refs)
    LOGGER.info(f"evaluation {img_num} images, {gt_cap_num} GT captions")
    return eval_log


@torch.no_grad()
def inference(model, eval_loader, max_seq_len, batch_size):
    """
    generate id sequences for all instances in evaluation dataset
    :param model:
    :param eval_loader:
    :return:
    """
    # max_seq_len = COCO_MAX_DECODING_LEN
    model.eval()
    # print('utils/caption_eval.py model', type(model))
    if hvd.rank() == 0:
        pbar = tqdm(total=len(eval_loader))
    else:
        pbar = NoOp()
    # predicted_matrix = []
    predicted_matrix = torch.zeros(len(eval_loader.dataset), COCO_MAX_DECODING_LEN-1, device=torch.device("cuda"), dtype=torch.int32)
    # print('utils/caption_eval.py predicted_matrix: ', predicted_matrix.shape)
    for k, batch in enumerate(eval_loader):
        # print(k)
        for i in range(max_seq_len):
            scores = model(batch, compute_loss=False) #  batch * max_decoding_len * vocab_size
            # greedy decoding
            # print('util/caption_eval.py scores:', scores.shape)
            predicted_id = scores.argmax(dim=-1) # batch * max_decoding_len
            # print('utils/caption_eval.py predicted_id', predicted_id)
            batch['input_ids'][:, 1:] = predicted_id[:, :-1] #
        batch_predicted_matrix = batch['input_ids'][:, 1:]  # no <s> # batch * max_decoding_len-1
        start_index = k * batch_size
        end_index = k*batch_size + batch_predicted_matrix.size(0)
        predicted_matrix[start_index:end_index,:] = batch_predicted_matrix
        # predicted_matrix += batch_predicted_matrix.tolist()
        pbar.update(1)
    model.train()
    pbar.close()
    return predicted_matrix

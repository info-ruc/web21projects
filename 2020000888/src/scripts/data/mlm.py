"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

MLM datasets
"""
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip

from .data import (DetectFeatTxtTokDataset, TxtTokLmdb,
                   pad_tensors, get_gather_index)

def random_word(tokens, vocab_range, mask):
    """
    Masking some random tokens for Language Model task with probabilities as in
        the original BERT paper.
    :param tokens: list of int, tokenized sentence.
    :param vocab_range: for choosing a random word [106,vocal-size]
    :param mask: self.txt_db.mask = 103.
    :return: (list of int, list of int), masked tokens and related labels for
        LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = mask

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(range(*vocab_range)))

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            output_label.append(token)
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
    if len(output_label)==0 and len(tokens)==0:
        return tokens, output_label
    if all(o == -1 for o in output_label):
        # at least mask 1
        output_label[0] = tokens[0]
        tokens[0] = mask

    return tokens, output_label


class MlmDataset(DetectFeatTxtTokDataset):
    def __init__(self, txt_db, img_db):
        assert isinstance(txt_db, TxtTokLmdb)
        super().__init__(txt_db, img_db)

    def __getitem__(self, i):
        """
        for one sample
        example['input_ids'] doesn't contain cls and sep, added in create_mlm_io()
        Return:
        - input_ids    : (L, ), i.e., [cls, wd, wd, ..., sep, 0, 0], 0s padded
        - img_feat     : (num_bb, d)
        - img_pos_feat : (num_bb, 7)
        - attn_masks   : (L + num_bb, ), ie., [1, 1, ..., 0, 0, 1, 1]
        - txt_labels   : (L, ), [-1, -1, wid, -1, -1, -1]
        0's padded so that (L + num_bb) % 8 == 0
        """
        example = super().__getitem__(i)
        # text input
        input_ids, txt_labels = self.create_mlm_io(example['input_ids'])
        # img input
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'])
        """if len(input_ids) == 2: # anwen hu 2020/12/15: debug null input ids
           print('null input_ids id', i)
           print(example['img_fname'])"""
        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)
        # lrc add for each modality.
        attn_masks_txt = torch.ones(len(input_ids), dtype=torch.long)
        attn_masks_img = torch.ones(num_bb, dtype=torch.long)

        return input_ids, img_feat, img_pos_feat, attn_masks, txt_labels, attn_masks_txt, attn_masks_img

    def create_mlm_io(self, input_ids):
        '''
        for one sample
        根据输入的文本的真实的非扩展后的input ids，执行随机遮蔽策略，
        得到遮蔽后的句子以及遮蔽的单词序列。
        original input_ids:[138, 1590, 2807, 1113, 170, 3119, 4073, 5205, 1380, 119] 
        noextend input_ids:[138, 1590, 2807, 1113, 103, 3119, 4073, 5205, 1380, 119] # 
        noextend txt_labels:[-1, -1, -1, -1, 170, -1, -1, -1, -1, -1]
        '''
        input_ids, txt_labels = random_word(input_ids,
                                            self.txt_db.v_range,
                                            self.txt_db.mask)
        # extend input_ids
        input_ids = torch.tensor([self.txt_db.cls_]
                                 + input_ids
                                 + [self.txt_db.sep])
        # extend txt_labels
        txt_labels = torch.tensor([-1] + txt_labels + [-1])
        return input_ids, txt_labels


def mlm_collate(inputs):
    """
    Return:
    n = batch-size
    :input_ids    (n, max_L) padded with 0
    :position_ids (n, max_L) padded with 0
    :txt_lens     list of [txt_len]
    :img_feat     (n, max_num_bb, feat_dim)
    :img_pos_feat (n, max_num_bb, 7)
    :num_bbs      list of [num_bb]
    :attn_masks   (n, max_{L + num_bb}) padded with 0
    :attn_masks_txt   (n, max_L) padded with 0
    :attn_masks_img   (n, max_num_bb) padded with 0
    :txt_labels   (n, max_L) padded with -1
    """
    (input_ids, img_feats, img_pos_feats, attn_masks, txt_labels, attn_masks_txt, attn_masks_img
     ) = map(list, unzip(inputs))

    # text batches
    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    txt_labels = pad_sequence(txt_labels, batch_first=True, padding_value=-1)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    # image batches
    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    # lrc add for two-flow
    attn_masks_txt = pad_sequence(attn_masks_txt, batch_first=True, padding_value=0)
    attn_masks_img = pad_sequence(attn_masks_img, batch_first=True, padding_value=0)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)  # = attn_masks.size()[1]
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'attn_masks_txt': attn_masks_txt,
             'attn_masks_img': attn_masks_img,
             'gather_index': gather_index,
             'txt_labels': txt_labels}
    return batch

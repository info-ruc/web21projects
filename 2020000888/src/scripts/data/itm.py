"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Itm dataset
"""
from collections import defaultdict
import copy
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from cytoolz import concat
import numpy as np

from .data import (DetectFeatTxtTokDataset, DetectFeatTxtTokTestDataset, DetectFeatLmdb, TxtTokLmdb,
                   pad_tensors, get_gather_index, get_ids_and_lens)
from .sampler import TokenBucketSampler


class TokenBucketSamplerForItm(TokenBucketSampler):
    def __init__(self, dset, *args, **kwargs):
        super().__init__(dset.lens, *args, **kwargs)
        self.dset = dset

    def __iter__(self):
        it = super().__iter__()
        self.dset.new_epoch()
        self._lens = self.dset.lens
        return it


def _has_overlap(la, lb):
    if len(la) < len(lb):
        la, lb = lb, la
    s = set(la)
    return any(b in s for b in lb)


def sample_negative(sample_pool, ground_truths, num_sample):
    """ random and retry """
    outputs = ground_truths[:1]
    while _has_overlap(outputs, ground_truths):
        outputs = random.sample(sample_pool, num_sample)
    return outputs


class ItmDataset(DetectFeatTxtTokDataset):
    # TODO: merge txt_db to decrease memory usage // 减少img_fname重叠
    """ NOTE this Dataset handles distributed training itself
    (for more efficient negative sampling) """
    def __init__(self, txt_db, img_db, neg_sample_p=0.5):
        assert isinstance(txt_db, TxtTokLmdb)
        assert isinstance(img_db, DetectFeatLmdb)

        self.txt_db = txt_db
        self.img_db = img_db

        self.txt_lens, self.ids = get_ids_and_lens(txt_db)
        #print('########### all_imgs #############')
        self.all_imgs = list(set(txt_db[id_]['img_fname'] for id_ in self.ids))
        self.all_imgs = sorted(self.all_imgs)
        #print('########### all_imgs_end #############')
        self.neg_sample_p = neg_sample_p
        #print('########### new epoch #############')
        self.new_epoch()
        #print('########### new epoch end #############')
        
    def new_epoch(self):
        """ should be called every epoch for more randomness"""
        self.labels = np.random.choice(
            [0, 1], size=len(self.ids),
            p=[self.neg_sample_p, 1-self.neg_sample_p])

        self.lens = []
        self.train_imgs = []
        for i, (id_, tl) in enumerate(zip(self.ids, self.txt_lens)):
            img_fname = super().__getitem__(i)['img_fname']
            if self.labels[i] == 0:
                img_fname = sample_negative(self.all_imgs, [img_fname], 1)[0]
            self.train_imgs.append(img_fname)
            self.lens.append(tl + self.img_db.name2nbb[img_fname])

    def __getitem__(self, i):
        ''' target 表示图文是否匹配
            attn_mask 表示该序列有多长 txt + imgs
        '''
        example = super().__getitem__(i)
        # labels and negative images should be sampled every epoch
        ground_truth_label = self.labels[i]
        img_fname = self.train_imgs[i]
        img_feat, img_pos_feat, num_bb = self._get_img_feat(img_fname)

        # text input
        input_ids = example['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)

        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)
        # lrc add
        attn_masks_txt = torch.ones(len(input_ids), dtype=torch.long)
        attn_masks_img = torch.ones(num_bb, dtype=torch.long)

        target = torch.Tensor(1).long()
        target.data.fill_(ground_truth_label)

        return input_ids, img_feat, img_pos_feat, attn_masks, target, attn_masks_txt, attn_masks_img

def itm_collate(inputs):
    ''' attn_mask 是文本和图像合起来做的padding
        gather_index
        input: [(input_ids, img_feat, img_pos_feat, attn_masks, target, attn_masks_txt, attn_masks_img), (), ()]
    '''
    (input_ids, img_feats, img_pos_feats, attn_masks, targets, attn_masks_txt, attn_masks_img
     ) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    # lrc add start
    attn_masks_txt = pad_sequence(attn_masks_txt, batch_first=True, padding_value=0)
    attn_masks_img = pad_sequence(attn_masks_img, batch_first=True, padding_value=0)

    targets = torch.cat(targets, dim=0)
    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'attn_masks_txt': attn_masks_txt,
             'attn_masks_img': attn_masks_img,
             'gather_index': gather_index,
             'targets': targets}
    return batch


def _compute_ot_scatter(txt_lens, max_txt_len, joint_len):
    ot_scatter = torch.arange(0, joint_len, dtype=torch.long
                              ).unsqueeze(0).repeat(len(txt_lens), 1)
    for i, tl in enumerate(txt_lens):
        max_ind = max_txt_len + (joint_len-tl)
        ot_scatter.data[i, tl:] = torch.arange(max_txt_len, max_ind,
                                               dtype=torch.long).data
    return ot_scatter


def _compute_pad(lens, max_len):
    pad = torch.zeros(len(lens), max_len, dtype=torch.bool)
    for i, l in enumerate(lens):
        pad.data[i, l:].fill_(1)
    return pad


def itm_ot_collate(inputs):
    (input_ids, img_feats, img_pos_feats, attn_masks, targets, attn_masks_txt, attn_masks_img
     ) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
     # lrc add start
    attn_masks_txt = pad_sequence(attn_masks_txt, batch_first=True, padding_value=0)
    attn_masks_img = pad_sequence(attn_masks_img, batch_first=True, padding_value=0)

    targets = torch.cat(targets, dim=0)
    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    # OT inputs
    max_tl = max(txt_lens)
    max_nbb = max(num_bbs)
    ot_scatter = _compute_ot_scatter(txt_lens, max_tl, attn_masks.size(1))
    txt_pad = _compute_pad(txt_lens, max_tl)
    img_pad = _compute_pad(num_bbs, max_nbb)
    ot_inputs = {'ot_scatter': ot_scatter,
                 'scatter_max': ot_scatter.max().item(),
                 'txt_pad': txt_pad,
                 'img_pad': img_pad}

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'attn_masks_txt': attn_masks_txt,
             'attn_masks_img': attn_masks_img,
             'gather_index': gather_index,
             'targets': targets,
             'ot_inputs': ot_inputs}

    return batch


class ItmRankDataset(DetectFeatTxtTokDataset):
    def __init__(self, txt_db, img_db, neg_sample_size=1):
        assert neg_sample_size > 0, \
            "ItmRankDataset need at least 1 negative sample"
        super().__init__(txt_db, img_db)

        txt2img = self.txt_db.txt2img
        self.txt2img = {id_: txt2img[id_] for id_ in self.ids}
        # images partitioned by rank
        self.img2txts = defaultdict(list)
        for id_, img in self.txt2img.items():
            self.img2txts[img].append(id_)
        self.img_name_list = list(self.img2txts.keys())

        assert neg_sample_size > 0
        self.neg_sample_size = neg_sample_size

    def __getitem__(self, i):
        gt_txt_id = self.ids[i]
        gt_img_fname = self.txt2img[gt_txt_id]

        id_pairs = [(gt_txt_id, gt_img_fname)]
        # sample negatives
        neg_sample_img_ids = sample_negative(
            self.img_name_list, [gt_img_fname], self.neg_sample_size)
        neg_sample_txt_ids = sample_negative(
            self.ids, self.img2txts[gt_img_fname], self.neg_sample_size)
        id_pairs.extend([(gt_txt_id, neg_img_id)
                         for neg_img_id in neg_sample_img_ids] +
                        [(neg_txt_id, gt_img_fname)
                         for neg_txt_id in neg_sample_txt_ids])
        inputs = self._collect_inputs(id_pairs)
        assert len(inputs) == (1 + 2*self.neg_sample_size)
        return inputs

    def _collect_inputs(self, id_pairs):
        # create input features
        inputs = []
        for txt_id, img_id in id_pairs:
            example = self.txt_db[txt_id]
            # text input
            input_ids = example['input_ids']
            input_ids = self.txt_db.combine_inputs(input_ids)
            # img input
            img_feat, img_pos_feat, num_bb = self._get_img_feat(img_id)
            # mask
            attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)
            # zyd add
            attn_masks_txt = torch.ones(len(input_ids), dtype=torch.long)
            attn_masks_img = torch.ones(num_bb, dtype=torch.long)

            inputs.append((input_ids, img_feat, img_pos_feat, attn_masks, attn_masks_txt, attn_masks_img))

        return inputs


def itm_rank_collate(inputs):
    (input_ids, img_feats, img_pos_feats, attn_masks, attn_masks_txt, attn_masks_img
     ) = map(list, unzip(concat(i for i in inputs)))

    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    # zyd add start
    attn_masks_txt = pad_sequence(attn_masks_txt, batch_first=True, padding_value=0)
    attn_masks_img = pad_sequence(attn_masks_img, batch_first=True, padding_value=0)

    sample_size = len(inputs[0])
    assert all(sample_size == len(i) for i in inputs)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'attn_masks_txt': attn_masks_txt,
             'attn_masks_img': attn_masks_img,
             'gather_index': gather_index,
             'sample_size': sample_size}
    return batch


def itm_rank_batch_collate(inputs):
    batch_size = len(inputs)
    (input_ids, img_feats, img_pos_feats, attn_masks, attn_masks_txt, attn_masks_img
     ) = map(list, unzip(concat(i for i in inputs)))

    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    max_txt_len = input_ids.size(-1)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    max_bbox = img_feat.size(1)
    feat_dim = img_feat.size(2)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)
    pos_feat_dim = img_pos_feat.size(2)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    max_len = attn_masks.size(-1)
    # zyd add start
    attn_masks_txt = pad_sequence(attn_masks_txt, batch_first=True, padding_value=0)
    attn_masks_img = pad_sequence(attn_masks_img, batch_first=True, padding_value=0)

    sample_size = len(inputs[0])
    assert all(sample_size == len(i) for i in inputs)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    #import pdb;pdb.set_trace()

    batch = {'input_ids': input_ids.view(batch_size, -1, max_txt_len),
             'position_ids': position_ids,
             'img_feat': img_feat.view(batch_size, -1, max_bbox, feat_dim),
             'img_pos_feat': img_pos_feat.view(batch_size, -1, max_bbox, pos_feat_dim),
             'attn_masks': attn_masks.view(batch_size, -1, max_len),
             'attn_masks_txt': attn_masks_txt.view(batch_size, -1, max_txt_len),
             'attn_masks_img': attn_masks_img.view(batch_size, -1, max_bbox),
             'gather_index': gather_index.view(batch_size, -1, out_size),
             'sample_size': sample_size}
    return batch

class ItmNCEDataset(DetectFeatTxtTokDataset):
    '''
    batch = one align txt-img pair and 2*neg_sample_size mis-align pairs
    '''
    def __init__(self, txt_db, img_db, neg_sample_size=1):
        assert neg_sample_size > 0, "need at least 1 negative sample"
        super().__init__(txt_db, img_db)

        txt2img = self.txt_db.txt2img
        self.txt2img = {id_: txt2img[id_] for id_ in self.ids}
        self.img2txts = self.txt_db.img2txts
        self.img_name_list = list(self.img2txts.keys())
        self.neg_sample_size = neg_sample_size

    def __getitem__(self, i):
        gt_txt_id = self.ids[i]
        gt_img_fname = self.txt2img[gt_txt_id]

        neg_sample_img_ids = sample_negative(
            self.img_name_list, [gt_img_fname], self.neg_sample_size)
        neg_sample_txt_ids = sample_negative(
            self.ids, self.img2txts[gt_img_fname], self.neg_sample_size)

        txt_ids = [gt_txt_id] + neg_sample_txt_ids + [gt_txt_id]*self.neg_sample_size
        img_ids = [gt_img_fname] + [gt_img_fname]*self.neg_sample_size + neg_sample_img_ids
        
        assert len(txt_ids) == len(img_ids) == 2*self.neg_sample_size+1

        batch_size = 2*self.neg_sample_size+1

        input_ids = []

        for txt_id in txt_ids:
            input_id = self.txt_db[txt_id]['input_ids']
            
            input_id = self.txt_db.combine_inputs(input_id)
            input_ids.append(input_id)

        txt_lens = [i.size(0) for i in input_ids]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long)
        position_ids = position_ids.unsqueeze(0)

        img_feats, img_pos_feats, num_bbs = map(
            list, unzip(map(self._get_img_feat, img_ids))
        )
        img_feat = pad_tensors(img_feats, num_bbs)
        img_pos_feats = pad_tensors(img_pos_feats, num_bbs)

        max_txt_len = max(txt_lens)
        max_bbs = max(num_bbs)

        assert len(num_bbs) == len(txt_lens) == batch_size
        max_txt_bbs = max([txt_lens[i]+num_bbs[i] for i in range(batch_size)])
        
        attn_masks = torch.zeros(batch_size, max_txt_bbs).long()
        for i, (txt_len, nbb) in enumerate(zip(txt_lens, num_bbs)):
            attn_masks.data[i, :txt_len+nbb].fill_(1)
        
        attn_masks_txt = torch.zeros(batch_size, max_txt_len).long()
        attn_masks_img = torch.zeros(batch_size, max_bbs).long()

        for i, nbb in enumerate(num_bbs):
            attn_masks_img.data[i, :nbb].fill_(1)
        for i, txt_len in enumerate(txt_lens):
            attn_masks_txt.data[i, :txt_len].fill_(1)
        
        out_size = attn_masks.size(1)
        gather_index = get_gather_index(txt_lens, num_bbs, batch_size, max_txt_len, out_size)

        batch = {
            'input_ids': input_ids,
            'position_ids': position_ids,
            'img_feat': img_feat,
            'img_pos_feat': img_pos_feats,
            'attn_masks': attn_masks,
            'attn_masks_txt': attn_masks_txt,
            'attn_masks_img': attn_masks_img,
            'gather_index': gather_index
        }
        return batch


def itm_nce_collate(inputs):
    assert len(inputs) == 1
    return inputs[0]


class ItmRankDatasetHardNegFromText(DetectFeatTxtTokDataset):
    '''
    文本作为query 去 sample 负例的 images
    '''
    def __init__(self, txt_db, img_db, neg_sample_size=1):
        assert neg_sample_size > 0, "need at least 1 negative sample"
        super().__init__(txt_db, img_db)

        txt2img = self.txt_db.txt2img
        self.txt2img = {id_: txt2img[id_] for id_ in self.ids}
        self.img2txts = self.txt_db.img2txts
        self.img_name_list = list(self.img2txts.keys())
        self.neg_sample_size = neg_sample_size

    def __getitem__(self, i):
        gt_txt_id = self.ids[i]
        gt_img_fname = self.txt2img[gt_txt_id]

        input_ids = self.txt_db[gt_txt_id]['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)
        input_ids = input_ids.unsqueeze(0)
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                    ).unsqueeze(0)

        neg_img_ids = sample_negative(
            self.img_name_list, [gt_img_fname], self.neg_sample_size)
        img_ids = [gt_img_fname] + neg_img_ids
        # process image features (gt always first)
        img_feats, img_pos_feats, num_bbs = map(
            list, unzip(map(self._get_img_feat, img_ids)))
        img_feat = pad_tensors(img_feats, num_bbs)
        img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

        tl = input_ids.size(1)
        attn_masks = torch.zeros(len(img_ids), max(num_bbs) + tl).long()
        for i, nbb in enumerate(num_bbs):
            attn_masks.data[i, :tl+nbb].fill_(1)
        
        # lrc add for two-flow
        attn_masks_txt = torch.ones(len(img_ids), tl).long()
        attn_masks_img = torch.zeros(len(img_ids), max(num_bbs)).long()
        for i, nbb in enumerate(num_bbs):
            attn_masks_img.data[i, :nbb].fill_(1)
        
        out_size = attn_masks.size(1)
        gather_index = get_gather_index([tl]*len(img_ids), num_bbs,
                                        len(img_ids), tl, out_size)

        batch = {'input_ids': input_ids,
                 'position_ids': position_ids,
                 'img_feat': img_feat,
                 'img_pos_feat': img_pos_feat,
                 'attn_masks': attn_masks,
                 'attn_masks_txt': attn_masks_txt, 
                 'attn_masks_img': attn_masks_img,
                 'gather_index': gather_index}
        return batch


class ItmRankDatasetHardNegFromImage(DetectFeatTxtTokDataset):
    '''
    Image 作为 query 去 sample 负例的 texts
    '''
    def __init__(self, txt_db, img_db, neg_sample_size=1):
        assert neg_sample_size > 0, "need at least 1 negative sample"
        super().__init__(txt_db, img_db)

        txt2img = self.txt_db.txt2img
        self.txt2img = {id_: txt2img[id_] for id_ in self.ids}
        self.img2txts = self.txt_db.img2txts
        self.txt_name_list = list(self.txt2img.keys())
        self.neg_sample_size = neg_sample_size

    def __getitem__(self, i):
        gt_txt_id = self.ids[i]
        gt_img_id = self.txt2img[gt_txt_id]
        gt_txt_ids = self.img2txts[gt_img_id]

        # process image features (gt always first)
        img_feat, img_pos_feat, nbb = self._get_img_feat(gt_img_id)
        img_feat = img_feat.unsqueeze(0)
        img_pos_feat = img_pos_feat.unsqueeze(0)

        # sample negative
        neg_txt_ids = sample_negative(
            self.txt_name_list, gt_txt_ids, self.neg_sample_size)
        txt_ids = [gt_txt_id] + neg_txt_ids

        # process text inputs
        all_inputs = []
        txt_lens = []
        for txt_id in txt_ids:
            input_ids = self.txt_db.combine_inputs(
                self.txt_db[txt_id]['input_ids'])
            all_inputs.append(input_ids)
            txt_lens.append(len(input_ids))
        input_ids = pad_sequence(all_inputs, batch_first=True, padding_value=0)
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                    ).unsqueeze(0)

        attn_masks = torch.zeros(len(txt_ids), max(txt_lens) + nbb).long()
        for i, tl in enumerate(txt_lens):
            attn_masks.data[i, :tl+nbb].fill_(1)
        # batch 内所有的 image 都是一样的
        attn_masks_img = torch.ones(len(txt_ids), nbb).long()
        attn_masks_txt = torch.zeros(len(txt_ids), max(txt_lens)).long()
        for i, tl in enumerate(txt_lens):
            attn_masks_txt.data[i, :tl].fill_(1)

        out_size = attn_masks.size(1)
        gather_index = get_gather_index(txt_lens, [nbb]*len(txt_ids),
                                        len(txt_ids), max(txt_lens), out_size)

        batch = {'input_ids': input_ids,
                 'position_ids': position_ids,
                 'img_feat': img_feat,
                 'img_pos_feat': img_pos_feat,
                 'attn_masks': attn_masks,
                 'attn_masks_txt': attn_masks_txt,
                 'attn_masks_img': attn_masks_img,
                 'gather_index': gather_index}
        return batch


def itm_rank_hn_collate(inputs):
    assert len(inputs) == 1
    return inputs[0]


class ItmValDataset(DetectFeatTxtTokDataset):
    """ For evaluating Image-Text-Retrieval task """
    def __init__(self, db_dir, img_dir, mini_batch_size=400):
        super().__init__(db_dir, img_dir)
        del self.lens
        self.txt2img = self.txt_db.txt2img
        self.img2txts = self.txt_db.img2txts
        self.all_img_ids = list(self.img2txts.keys())

        assert len(self.img2txts) >= mini_batch_size > 0
        self.bs = mini_batch_size

    def _get_batch_ids(self, i):
        gt_txt_id = self.ids[i]
        gt_img_id = self.txt2img[gt_txt_id]

        # sample fixed negatives for each gt image
        i = self.all_img_ids.index(gt_img_id)
        neg_st = i+1
        neg_end = neg_st+self.bs-1
        if neg_end > len(self.all_img_ids):
            # warp around
            neg_end -= len(self.all_img_ids)
            neg_img_ids = (self.all_img_ids[neg_st:]
                           + self.all_img_ids[:neg_end])
        else:
            neg_img_ids = self.all_img_ids[neg_st:neg_end]

        assert len(neg_img_ids) == (self.bs - 1),\
            "Did not sample enough neg samples"

        return gt_img_id, neg_img_ids

    def __getitem__(self, i):
        """ this returns list of mini-batches """
        gt_img_id, neg_img_ids = self._get_batch_ids(i)
        # NOTE 1st one is gt img
        batch = self.get_batch(i, [gt_img_id] + neg_img_ids)
        return batch

    def get_batch(self, i, img_ids):
        example = super().__getitem__(i)

        input_ids = example['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)
        input_ids = input_ids.unsqueeze(0).expand(len(img_ids), -1).clone()
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                    ).unsqueeze(0)

        # process image features (gt always first)
        img_feats, img_pos_feats, num_bbs = map(
            list, unzip(map(self._get_img_feat, img_ids)))
        img_feat = pad_tensors(img_feats, num_bbs)
        img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

        tl = input_ids.size(1)
        attn_masks = torch.zeros(len(img_ids), max(num_bbs) + tl).long()
        for i, nbb in enumerate(num_bbs):
            attn_masks.data[i, :tl+nbb].fill_(1)
        # lrc add for two-flow
        attn_masks_txt = torch.ones(len(img_ids), tl).long()
        attn_masks_img = torch.zeros(len(img_ids), max(num_bbs)).long()
        for i, nbb in enumerate(num_bbs):
            attn_masks_img.data[i, :nbb].fill_(1)
        
        out_size = attn_masks.size(1)
        gather_index = get_gather_index([tl]*len(img_ids), num_bbs,
                                        len(img_ids), tl, out_size)
        batch = {'input_ids': input_ids,
                 'position_ids': position_ids,
                 'img_feat': img_feat,
                 'img_pos_feat': img_pos_feat,
                 'attn_masks': attn_masks,
                 'attn_masks_txt': attn_masks_txt, 
                 'attn_masks_img': attn_masks_img,
                 'gather_index': gather_index}
        return batch


def itm_val_collate(inputs):
    assert len(inputs) == 1, "input batch size > 1"
    return inputs[0]


class ItmEvalDataset(ItmValDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_img_ids = sorted(copy.deepcopy(self.all_img_ids),
                                  key=lambda i: self.img_db.name2nbb[i])

    def __getitem__(self, i):
        mini_batches = []
        for st in range(0, len(self.all_img_ids), self.bs):
            mini_batches.append(
                self.get_batch(i, self.all_img_ids[st:st+self.bs]))
        return mini_batches

class ItmTestDataset(DetectFeatTxtTokTestDataset):
    def __init__(self, db_dir, img_dir, mini_batch_size=400, start_idx=0, end_idx=None):
        super().__init__(db_dir, img_dir)
        self.all_img_ids = list(self.img_db.name2nbb.keys())
        self.bs = min(len(self.all_img_ids), mini_batch_size)
        self.all_img_ids = sorted(copy.deepcopy(self.all_img_ids),
                                  key=lambda i: self.img_db.name2nbb[i])
        self.all_img_ids = self.all_img_ids[start_idx: end_idx]

    def get_input_ids(self, i):
        example = super().__getitem__(i)
        return example['input_ids']
    
    def get_total_input_ids(self, i):
        _id_keys = list(self.txt_db.id2len.keys())
        _id = _id_keys[i]
        example = self.txt_db[_id]
        return example['input_ids']

    def get_batch(self, i, img_ids):
        example = super().__getitem__(i)

        input_ids = example['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)
        input_ids = input_ids.unsqueeze(0).expand(len(img_ids), -1).clone()
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                    ).unsqueeze(0)

        # process image features (gt always first)
        img_feats, img_pos_feats, num_bbs = map(
            list, unzip(map(self._get_img_feat, img_ids)))
        img_feat = pad_tensors(img_feats, num_bbs)
        img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

        tl = input_ids.size(1)
        attn_masks = torch.zeros(len(img_ids), max(num_bbs) + tl).long()
        for i, nbb in enumerate(num_bbs):
            attn_masks.data[i, :tl+nbb].fill_(1)
        # lrc add for two-flow
        attn_masks_txt = torch.ones(len(img_ids), tl).long()
        attn_masks_img = torch.zeros(len(img_ids), max(num_bbs)).long()
        for i, nbb in enumerate(num_bbs):
            attn_masks_img.data[i, :nbb].fill_(1)
        
        out_size = attn_masks.size(1)
        gather_index = get_gather_index([tl]*len(img_ids), num_bbs,
                                        len(img_ids), tl, out_size)
        batch = {'input_ids': input_ids,
                 'position_ids': position_ids,
                 'img_feat': img_feat,
                 'img_pos_feat': img_pos_feat,
                 'attn_masks': attn_masks,
                 'attn_masks_txt': attn_masks_txt, 
                 'attn_masks_img': attn_masks_img,
                 'gather_index': gather_index}
        return batch

    def __getitem__(self, i):
        mini_batches = []
        for st in range(0, len(self.all_img_ids), self.bs):
            mini_batches.append(
                self.get_batch(i, self.all_img_ids[st:st+self.bs]))
        return mini_batches

class ItmACCRankDataset(DetectFeatTxtTokTestDataset):
    def __init__(self, db_dir, img_dir, mini_batch_size=400, img_id="0", end_idx=None):
        super().__init__(db_dir, img_dir)
        self.all_img_ids = list(self.img_db.name2nbb.keys())
        self.bs = min(len(self.all_img_ids), mini_batch_size)
        self.all_img_ids = sorted(copy.deepcopy(self.all_img_ids),
                                  key=lambda i: self.img_db.name2nbb[i])
                                  
        self.all_img_ids = [img_id+'.npz']# self.all_img_ids[start_idx: end_idx]

    def get_input_ids(self, i):
        example = super().__getitem__(i)
        return example['input_ids']
    
    def get_total_input_ids(self, i):
        _id_keys = list(self.txt_db.id2len.keys())
        _id = _id_keys[i]
        example = self.txt_db[_id]
        return example['input_ids']

    def get_batch(self, i, img_ids):
        example = super().__getitem__(i)

        input_ids = example['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)
        input_ids = input_ids.unsqueeze(0).expand(len(img_ids), -1).clone()
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                    ).unsqueeze(0)

        # process image features (gt always first)
        img_feats, img_pos_feats, num_bbs = map(
            list, unzip(map(self._get_img_feat, img_ids)))
        img_feat = pad_tensors(img_feats, num_bbs)
        img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

        tl = input_ids.size(1)
        attn_masks = torch.zeros(len(img_ids), max(num_bbs) + tl).long()
        for i, nbb in enumerate(num_bbs):
            attn_masks.data[i, :tl+nbb].fill_(1)
        # lrc add for two-flow
        attn_masks_txt = torch.ones(len(img_ids), tl).long()
        attn_masks_img = torch.zeros(len(img_ids), max(num_bbs)).long()
        for i, nbb in enumerate(num_bbs):
            attn_masks_img.data[i, :nbb].fill_(1)
        
        out_size = attn_masks.size(1)
        gather_index = get_gather_index([tl]*len(img_ids), num_bbs,
                                        len(img_ids), tl, out_size)
        batch = {'input_ids': input_ids,
                 'position_ids': position_ids,
                 'img_feat': img_feat,
                 'img_pos_feat': img_pos_feat,
                 'attn_masks': attn_masks,
                 'attn_masks_txt': attn_masks_txt, 
                 'attn_masks_img': attn_masks_img,
                 'gather_index': gather_index}
        return batch

    def __getitem__(self, i):
        mini_batches = []
        for st in range(0, len(self.all_img_ids), self.bs):
            mini_batches.append(
                self.get_batch(i, self.all_img_ids[st:st+self.bs]))
        return mini_batches

class ItmTestImg2TxtDataset(DetectFeatTxtTokTestDataset):
    def __init__(self, db_dir, img_dir):
        super().__init__(db_dir, img_dir)
        self.all_img_ids = list(self.img_db.name2nbb.keys())
        self.bs = len(self.all_img_ids)
        self.all_img_ids = sorted(copy.deepcopy(self.all_img_ids),
                                  key=lambda i: self.img_db.name2nbb[i])
        self.all_img_ids = self.all_img_ids

    def get_input_ids(self, i):
        example = super().__getitem__(i)
        return example['input_ids']

    def __getitem__(self, i):
        example = super().__getitem__(i)

        input_ids = example['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)
        input_ids = input_ids.unsqueeze(0).expand(len(self.all_img_ids), -1).clone()
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                    ).unsqueeze(0)

        # process image features (gt always first)
        img_feats, img_pos_feats, num_bbs = map(
            list, unzip(map(self._get_img_feat, self.all_img_ids)))
        img_feat = pad_tensors(img_feats, num_bbs)
        img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

        tl = input_ids.size(1)
        attn_masks = torch.zeros(len(self.all_img_ids), max(num_bbs) + tl).long()
        for i, nbb in enumerate(num_bbs):
            attn_masks.data[i, :tl+nbb].fill_(1)
        # lrc add for two-flow
        attn_masks_txt = torch.ones(len(self.all_img_ids), tl).long()
        attn_masks_img = torch.zeros(len(self.all_img_ids), max(num_bbs)).long()
        for i, nbb in enumerate(num_bbs):
            attn_masks_img.data[i, :nbb].fill_(1)
        
        out_size = attn_masks.size(1)
        gather_index = get_gather_index([tl]*len(self.all_img_ids), num_bbs,
                                        len(self.all_img_ids), tl, out_size)
        batch = {'input_ids': input_ids,
                 'position_ids': position_ids,
                 'img_feat': img_feat,
                 'img_pos_feat': img_pos_feat,
                 'attn_masks': attn_masks,
                 'attn_masks_txt': attn_masks_txt, 
                 'attn_masks_img': attn_masks_img,
                 'gather_index': gather_index}
        return batch
        

itm_eval_collate = itm_val_collate

# anwen hu 2020/12/15
class EnItmDataset(DetectFeatTxtTokDataset):
    """
    load data for en_item task, no negative img sampled when fetch data, and no target
    """
    def __init__(self, txt_db, img_db):
        assert isinstance(txt_db, TxtTokLmdb)
        assert isinstance(img_db, DetectFeatLmdb)

        self.txt_db = txt_db
        self.img_db = img_db
        self.txt2img = self.txt_db.txt2img  # key:_id, value:img_fname
        self.img2txts = self.txt_db.img2txts  # key:img_fname value:list(id_)
        self.all_img_ids = list(self.img2txts.keys())

        self.txt_lens, self.ids = get_ids_and_lens(txt_db)
        # self.all_imgs = list(set(txt_db[id_]['img_fname'] for id_ in self.ids))
        self.lens = []
        self.train_imgs = []
        for i, (id_, tl) in enumerate(zip(self.ids, self.txt_lens)):
            img_fname = super().__getitem__(i)['img_fname']
            self.train_imgs.append(img_fname)
            self.lens.append(tl + self.img_db.name2nbb[img_fname])

    def __getitem__(self, i):
        ''' target 表示图文是否匹配
            attn_mask 表示该序列有多长 txt + imgs
        '''
        id_ = self.ids[i] #
        example = self.txt_db[id_]
        # labels and negative images should be sampled every epoch
        img_fname = self.train_imgs[i]
        img_feat, img_pos_feat, num_bb = self._get_img_feat(img_fname)

        # text input
        input_ids = example['input_ids']

        # modified by zyd 2021.1.3
        input_ids = torch.tensor(input_ids)
        # input_ids = self.txt_db.combine_inputs(input_ids)

        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)
        # lrc add
        attn_masks_txt = torch.ones(len(input_ids), dtype=torch.long)
        attn_masks_img = torch.ones(num_bb, dtype=torch.long)

        return input_ids, img_feat, img_pos_feat, attn_masks, attn_masks_txt, attn_masks_img, id_, img_fname


def enitm_collate(inputs):
    ''' attn_mask 是文本和图像合起来做的padding
        gather_index
        input: [(input_ids, img_feat, img_pos_feat, attn_masks, attn_masks_txt, attn_masks_img, img_fname), (), ()]
    '''
    # anwen hu 2020/12/17 add ids_ and img_fnames to save the txt and img encoder outputs
    (input_ids, img_feats, img_pos_feats, attn_masks, attn_masks_txt, attn_masks_img, ids_, img_fnames) = \
        map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    # lrc add start
    attn_masks_txt = pad_sequence(attn_masks_txt, batch_first=True, padding_value=0)
    attn_masks_img = pad_sequence(attn_masks_img, batch_first=True, padding_value=0)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'attn_masks_txt': attn_masks_txt,
             'attn_masks_img': attn_masks_img,
             'gather_index': gather_index,
             'targets': None,
             'ids_': ids_,
             'img_fnames':img_fnames}
    return batch

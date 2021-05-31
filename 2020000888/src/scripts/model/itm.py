"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

UNITER for ITM model
"""
from collections import defaultdict
import random

import torch
from torch import nn
from torch.nn import functional as F
from .model import UniterPreTrainedModel, UniterModel, UniterTwoFlowModel, UniterTwoFlowRetrievalModel
from .layer import l2norm
from data.data import get_gather_index
import pdb

class UniterForImageTextRetrieval(UniterPreTrainedModel):
    """ Finetune UNITER for image text retrieval
    """
    def __init__(self, config, img_dim, margin=0.2):
        super().__init__(config)
        # self.uniter = UniterModel(config, img_dim)
        # anwen hu 2020/12/20: use one-flow or two-flow model
        if config.model_mode == 'two_flow':
            print('Use the two flow model')
            self.two_flow = True
            self.fix_encoder = False if not hasattr(config, 'fix_encoder') else config.fix_encoder
            self.uniter = UniterTwoFlowModel(config, img_dim)
        else:
            self.two_flow = False
            print('Use the one flow model')
            self.uniter = UniterModel(config, img_dim)
        self.itm_output = nn.Linear(config.hidden_size, 2)
        self.rank_output = nn.Linear(config.hidden_size, 1)
        self.margin = margin
        self.apply(self.init_weights)
        self.loss_type = 'triplet' if not hasattr(config, 'loss_type') else config.loss_type

    def init_output(self):
        """ need to be called after from pretrained """
        self.rank_output.weight.data = self.itm_output.weight.data[1:, :]
        self.rank_output.bias.data = self.itm_output.bias.data[1:]

    def forward(self, batch, compute_loss=True, loss_type='triplet'):
        batch = defaultdict(lambda: None, batch)
        if self.two_flow:
            _, _, sequence_output = self.uniter(batch, output_all_encoded_layers=False, fix_encoder=self.fix_encoder)
        else:
            sequence_output = self.uniter(batch, output_all_encoded_layers=False)
        pooled_output = self.uniter.pooler(sequence_output)
        if loss_type == 'bce' and self.training:
            itm_scores = self.itm_output(pooled_output)
        else:
            rank_scores = self.rank_output(pooled_output)  # batch_size * (1+2*negative_num) * 1
        
        if compute_loss:
            if loss_type == 'triplet':
                # triplet loss
                # import pdb;pdb.set_trace()
                rank_scores_sigmoid = torch.sigmoid(rank_scores)
                sample_size = batch['sample_size']  # 1+2*negative_num
                scores = rank_scores_sigmoid.contiguous().view(-1, sample_size) # batch_size * (1+2*negative_num)
                pos = scores[:, :1]  # batch * 1
                neg = scores[:, 1:]  # batch * 2*negative_num
                rank_loss = torch.clamp(self.margin + neg - pos, 0)
                return rank_loss
            elif loss_type == 'contrastive':
                # nce loss
                # the first batch is positive
                loss = -F.log_softmax(rank_scores, dim=0)[0]
                return loss
            elif loss_type == 'bce':
                targets = batch['targets']
                loss = F.cross_entropy(itm_scores, targets, reduction='none')
                # import pdb;pdb.set_trace()
                return loss
            else:
                raise ValueError()
        else:
            return rank_scores

    # add by zhangliang 2012/1/14
    def forward_encoder(self, batch):
        assert self.two_flow
        txt_pooled_output, img_pooled_output, txt_encoded_layers, img_encoded_layers = self.uniter(batch, output_all_encoded_layers=False, fusion=False, encoder_out=True)
        scores = torch.mm(txt_pooled_output, img_pooled_output.t())
        rank_scores = scores.diag()
        return rank_scores, txt_encoded_layers, img_encoded_layers
    

    # add by zhangliang 2012/1/14
    def forward_fusion(self, batch, compute_loss=True, loss_type='triplet'):

        assert self.two_flow
        sequence_output = self.uniter.forward_fusion(batch)
        pooled_output = self.uniter.pooler(sequence_output)
        rank_scores = self.rank_output(pooled_output)
        if compute_loss:
            if loss_type == 'triplet':
                # triplet loss
                # import pdb;pdb.set_trace()
                rank_scores_sigmoid = torch.sigmoid(rank_scores)
                sample_size = batch['sample_size']  # 1+2*negative_num
                scores = rank_scores_sigmoid.contiguous().view(-1, sample_size) # batch_size * (1+2*negative_num)
                pos = scores[:, :1]  # batch * 1
                neg = scores[:, 1:]  # batch * 2*negative_num
                rank_loss = torch.clamp(self.margin + neg - pos, 0)
                return rank_loss
            elif loss_type == 'contrastive':
                # nce loss
                # the first batch is positive
                loss = -F.log_softmax(rank_scores, dim=0)[0]
                return loss
            else:
                raise ValueError()
        else:
            return rank_scores

class UniterForImageTextRetrievalHardNeg(UniterForImageTextRetrieval):
    """ Finetune UNITER for image text retrieval
    """
    def __init__(self, config, img_dim, margin=0.2, hard_size=16):
        super().__init__(config, img_dim, margin)
        self.hard_size = hard_size

    def forward(self, batch, sample_from='t', compute_loss=True, loss_type='triplet', select_hard='fusion'):
        # expect same input_ids for all pairs
        batch_size = batch['attn_masks'].size(0)
        input_ids = batch['input_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        if sample_from == 't':
            if input_ids.size(0) == 1:
                batch['input_ids'] = input_ids.expand(batch_size, -1)
        elif sample_from == 'i':
            if img_feat.size(0) == 1:
                batch['img_feat'] = img_feat.expand(batch_size, -1, -1)
            if img_pos_feat.size(0) == 1:
                batch['img_pos_feat'] = img_pos_feat.expand(batch_size, -1, -1)
        else:
            raise ValueError()

        if self.training and compute_loss:
            with torch.no_grad():
                self.eval()
                if select_hard == 'fusion':
                    scores = super().forward(batch, compute_loss=False)
                elif select_hard == 'encoder':
                    scores, txt_enc_feats, img_enc_feats = super().forward_encoder(batch)
                    batch['txt_emb'], batch['img_emb'] = txt_enc_feats, img_enc_feats
                else:
                    raise ValueError()
                hard_batch = self._get_hard_batch(batch, scores, sample_from)
                self.train()
            if select_hard == 'fusion':
                return super().forward(hard_batch, compute_loss=True, loss_type=loss_type)
            elif select_hard == 'encoder':
                return super().forward_fusion(hard_batch, compute_loss=True, loss_type=loss_type)
        else:
            return super().forward(batch, compute_loss)

    def _get_hard_batch(self, batch, scores, sample_from='t'):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attention_mask = batch['attn_masks']
        attn_masks_txt = batch['attn_masks_txt']
        attn_masks_img = batch['attn_masks_img']
        gather_index = batch['gather_index']
        txt_emb = batch['txt_emb']
        img_emb = batch['img_emb']
        hard_batch = {'sample_size': self.hard_size + 1}

        # NOTE first example is positive
        hard_indices = scores.squeeze(-1)[1:].topk(
            self.hard_size, sorted=False)[1] + 1
        indices = torch.cat([torch.zeros(1, dtype=torch.long,
                                         device=hard_indices.device),
                             hard_indices])

        attention_mask = attention_mask.index_select(0, indices)
        gather_index = gather_index.index_select(0, indices)
        
        # add by zyd 2020.12.28
        attn_masks_txt = attn_masks_txt.index_select(0, indices)
        attn_masks_img = attn_masks_img.index_select(0, indices)

        if position_ids.size(0) != 1:
            position_ids = position_ids[:self.hard_size+1]
        
        img_len = attn_masks_img.size(1)

        if sample_from == 't':
            # cut to minimum padding
            max_len = attention_mask.sum(dim=1).max().item()
            max_i = max_len - input_ids.size(1)
            attention_mask = attention_mask[:, :max_len]

            # add by zyd 2020.12.28
            attn_masks_img = attn_masks_img[:, :max_i]

            gather_index = gather_index[:, :max_len]
            img_feat = img_feat.index_select(0, indices)[:, :max_i, :]
            img_pos_feat = img_pos_feat.index_select(0, indices)[:, :max_i, :]
            
            # expect same input_ids for all pairs
            input_ids = input_ids[:self.hard_size+1]
            if img_emb is not None:
                img_emb = img_emb.index_select(0, indices)[:, :max_i, :]
            if txt_emb is not None:
                txt_emb = txt_emb.index_select(0, indices)[:self.hard_size+1]
        elif sample_from == 'i':
            input_ids = input_ids.index_select(0, indices)
            # expect same image features for all pairs
            img_feat = img_feat[:self.hard_size+1]
            img_pos_feat = img_pos_feat[:self.hard_size+1]
            if img_emb is not None:
                img_emb = img_emb.index_select(0, indices)[:self.hard_size+1]
            if txt_emb is not None:
                txt_emb = txt_emb.index_select(0, indices)[:self.hard_size+1]
        else:
            raise ValueError()

        hard_batch['input_ids'] = input_ids
        hard_batch['position_ids'] = position_ids
        hard_batch['img_feat'] = img_feat
        hard_batch['img_pos_feat'] = img_pos_feat
        hard_batch['attn_masks'] = attention_mask
        hard_batch['attn_masks_txt'] = attn_masks_txt
        hard_batch['attn_masks_img'] = attn_masks_img
        hard_batch['gather_index'] = gather_index
        hard_batch['txt_emb'] = txt_emb
        hard_batch['img_emb'] = img_emb
        hard_batch['img_emb'] = img_emb
        hard_batch['img_len'] = img_len
        return hard_batch


class UniterForTwoFlowEncoderImageTextRetrieval(UniterPreTrainedModel):
    """ Anwen Hu:2020/12/17: Finetune or Inference TWO FLOW UNITER encoder for image text retrieval
    """
    def __init__(self, config, img_dim, margin=0.2):
        super().__init__(config)
        assert config.model_mode == 'two_flow'
        self.uniter = UniterTwoFlowModel(config, img_dim)
        self.margin = margin  # only used for triple loss
        self.apply(self.init_weights)

    def forward(self, batch, compute_loss=True, loss_type='contrastive'):
        batch = defaultdict(lambda: None, batch)
        txt_encoded_output, img_encoded_output, sequence_output = self.uniter(batch, output_all_encoded_layers=False,
                                                                              fusion=False)
        # if contrastive loss, shape of txt/img_pooled_output is batch * dim
        # if triple loss, loss of txt/img_pooled_output is (batch_size * (1+2*negative_num)) * dim
        
        # modified by zyd 2020.12.31
        txt_pooled_output = txt_encoded_output
        img_pooled_output = img_encoded_output
        # txt_pooled_output = torch.mean(txt_encoded_output, dim=1)
        # img_pooled_output = torch.mean(img_encoded_output, dim=1)  # batch * dim

        # print('model/itm.py txt_pooled_output.shape', txt_pooled_output.shape)
        # print('model/itm.py img_pooled_output.shape', img_pooled_output.shape)
        if compute_loss:
            if loss_type == 'contrastive':
                # dot product: calculate similarity of each txt and img in the batch
                scores = torch.mm(txt_pooled_output, img_pooled_output.t())  # batch * batch
                # contrastive loss: for each txt, use negative pair to estimate the probability of positive pair
                txt2img_loss = -F.log_softmax(scores / 0.07, dim=1).diag()  # batch ; retrieve image
                img2txt_loss = -F.log_softmax(scores / 0.07, dim=0).diag()  # batch ; retrieve txt
                en_itm_loss = txt2img_loss + img2txt_loss  # reduce the loss, enforce the pos_prob to 1
                return en_itm_loss
            elif loss_type == 'triplet':
                # mul + sum  = dot
                rank_scores = torch.sum(torch.mul(txt_pooled_output, img_pooled_output), dim=-1)  # (batch_size * (1+2*negative_num))
                sample_size = batch['sample_size']  # 1+2*negative_num
                scores = rank_scores.contiguous().view(-1, sample_size)  # batch_size * (1+2*negative_num)
                pos = scores[:, :1]  # batch * 1
                neg = scores[:, 1:]  # batch * 2*negative_num
                rank_loss = torch.clamp(self.margin + neg - pos, 0)
                return rank_loss
            else:
                raise ValueError(f'Undefined loss: {loss_type}')

        else:
            return txt_pooled_output, img_pooled_output


# add by Zhangliang 2021/1/18
class UniterForITMEnITM(UniterPreTrainedModel):
    """ Liang Zhang:2021/1/18: Finetune or Inference TWO FLOW UNITER encoder and fusion together
    """
    def __init__(self, config, img_dim, margin=0.2):
        super().__init__(config)
        assert config.model_mode == 'two_flow'
        self.uniter = UniterTwoFlowModel(config, img_dim)
        self.margin = margin  # only used for triple loss
        self.apply(self.init_weights)

    def forward(self, batch, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        txt_pooled_output, img_pooled_output, txt_encoded_layers, img_encoded_layers = self.uniter(batch, output_all_encoded_layers=False, fusion=False, encoder_out=True)
        
        itm_batch = self._get_itm_batch(batch, txt_encoded_layers, img_encoded_layers, 31)
        # if contrastive loss, shape of txt/img_pooled_output is batch * dim
        # if triple loss, loss of txt/img_pooled_output is (batch_size * (1+2*negative_num)) * dim

        # txt_pooled_output = torch.mean(txt_encoded_output, dim=1)
        # img_pooled_output = torch.mean(img_encoded_output, dim=1)  # batch * dim

        # print('model/itm.py txt_pooled_output.shape', txt_pooled_output.shape)
        # print('model/itm.py img_pooled_output.shape', img_pooled_output.shape)
        if compute_loss:
            # dot product: calculate similarity of each txt and img in the batch
            scores = torch.mm(txt_pooled_output, img_pooled_output.t())  # batch * batch
            # contrastive loss: for each txt, use negative pair to estimate the probability of positive pair
            txt2img_loss = -F.log_softmax(scores / 0.07, dim=1).diag()  # batch ; retrieve image
            img2txt_loss = -F.log_softmax(scores / 0.07, dim=0).diag()  # batch ; retrieve txt
            en_itm_loss = txt2img_loss + img2txt_loss  # reduce the loss, enforce the pos_prob to 1



        else:
            return txt_pooled_output, img_pooled_output
    
    def _get_itm_batch(batch, txt_embeds, img_embeds, i, negative_size=15):    
        itm_batch = {}
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attn_masks = batch['attn_masks']
        gather_index = batch['gather_index']

        batch_size = input_ids.size(0)
        
        indices = list(range(batch_size))
        assert 2*negative_size < batch_size

        indices.remove(i)
        indices = random.sample(indices, 2*negative_size)


class UniterForTwoFlowImageTextRetrieval(UniterPreTrainedModel):
    """ Yida Zhao:2021/1/4: Train TWO FLOW UNITER encoder & fusion for image text retrieval
    """
    def __init__(self, config, img_dim, margin=0.2, hard_size=10, pos_size=8):
        super().__init__(config)
        assert config.model_mode == 'two_flow'
        self.eval_mode = config.eval_mode
        self.uniter = UniterTwoFlowRetrievalModel(config, img_dim)
        self.margin = margin  # only used for triple loss
        self.pos_size = pos_size
        self.hard_size = hard_size
        # self.itm_output = nn.Linear(config.hidden_size, 2)
        self.rank_output = nn.Linear(config.hidden_size, 1)
        self.activation = nn.Sigmoid()
        self.apply(self.init_weights)
        self.count = 0

    def forward(self, batch, compute_loss=True, loss_type='contrastive'):
        batch = defaultdict(lambda: None, batch)
        if self.eval_mode == 'fusion':
            fusion = True
        else:
            fusion = False
        txt_pooled_output, img_pooled_output, pooled_output = self.uniter(batch, output_all_encoded_layers=False,
                                                                              fusion=fusion)
        # if contrastive loss, shape of txt/img_pooled_output is batch * dim
        # if triple loss, loss of txt/img_pooled_output is (batch_size * (1+2*negative_num)) * dim

        if compute_loss:
            if loss_type == 'contrastive':
                # with torch.no_grad():
                #     self.eval()
                #     scores = super().forward(batch, compute_loss=False)
                #     hard_batch = self._get_hard_batch(batch, scores, sample_from)
                #     self.train()

                # dot product: calculate similarity of each txt and img in the batch
                scores = torch.mm(txt_pooled_output, img_pooled_output.t())  # batch * batch
                # contrastive loss: for each txt, use negative pair to estimate the probability of positive pair
                txt2img_loss = -F.log_softmax(scores / 0.07, dim=1).diag()  # batch ; retrieve image
                img2txt_loss = -F.log_softmax(scores / 0.07, dim=0).diag()  # batch ; retrieve txt
                en_itm_loss = txt2img_loss + img2txt_loss  # reduce the loss, enforce the pos_prob to 1

                # itm loss
                hard_batch = self._get_hard_batch(batch, scores)
                _, _, pooled_output = self.uniter(hard_batch, output_all_encoded_layers=False)

                # fusion itm
                itm_scores = self.activation(self.rank_output(pooled_output))
                itm_scores = itm_scores.reshape(self.pos_size, -1)
                pos = itm_scores[:, :1]  # batch * 1
                neg = itm_scores[:, 1:]  # batch * negative_num
                itm_loss = torch.clamp(self.margin + neg - pos, 0)
                # itm_loss = -F.log_softmax(itm_scores / 0.07, dim=1)[:, 1]
                # itm_loss = F.cross_entropy(itm_scores / 0.07, hard_batch['targets'], reduction='none')
                # return itm_loss.mean()
                en_itm_loss = en_itm_loss.mean()
                itm_loss = itm_loss.mean()
                self.count += 1
                if not self.count % 100:
                    print(en_itm_loss.item(), itm_loss.item())
                return itm_loss
                return en_itm_loss + itm_loss
            else:
                raise ValueError(f'Undefined loss: {loss_type}')

        else:
            if self.eval_mode == 'fusion':
                itm_scores = self.activation(self.rank_output(pooled_output))
                return itm_scores
            else:
                return txt_pooled_output, img_pooled_output
    
    def init_output(self):
        """ need to be called after from pretrained """
        self.rank_output.weight.data = self.itm_output.weight.data[1:, :]
        self.rank_output.bias.data = self.itm_output.bias.data[1:]

    def _get_hard_batch(self, batch, scores, sample_from='t'):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        # attention_mask = batch['attn_masks']
        attn_masks_txt = batch['attn_masks_txt']
        attn_masks_img = batch['attn_masks_img']
        # gather_index = batch['gather_index']
        hard_batch = {'sample_size': self.hard_size * self.pos_size * 2 + self.pos_size}
        targets = torch.zeros(hard_batch['sample_size'], dtype=torch.long, device=input_ids.device)
        targets[:self.pos_size] = 1

        # mask to clear diagonals which are positive pairs
        mask = torch.eye(scores.size(0)).byte().to(scores.device)
        scores = scores.masked_fill_(mask, -1e4)

        pos_indices = torch.tensor(random.sample(range(scores.size(0)), self.pos_size), device=scores.device)

        # sample from img
        img_hard_indices = scores.topk(self.hard_size, dim=1, sorted=False)[1]
        img_hard_indices = img_hard_indices[pos_indices]
        img_indices = torch.cat([pos_indices.unsqueeze(1),
                                 img_hard_indices,
                                 pos_indices.unsqueeze(1).repeat(1, self.hard_size)
                                ], dim=1).reshape(-1)
        txt_hard_indices = scores.t().topk(self.hard_size, dim=1, sorted=False)[1]
        txt_hard_indices = txt_hard_indices[pos_indices]
        txt_indices = torch.cat([pos_indices.unsqueeze(1).repeat(1, 1 + self.hard_size),
                                 txt_hard_indices
                                ], dim=1).reshape(-1)

        attn_masks_txt = attn_masks_txt[txt_indices]
        attn_masks_img = attn_masks_img[img_indices]
        txt_lens = attn_masks_txt.sum(dim=1)
        num_bbs = attn_masks_img.sum(dim=1)
        max_len_txt = txt_lens.max().item()
        max_len_img = num_bbs.max().item()
        attn_masks_txt = attn_masks_txt[:, :max_len_txt]
        attn_masks_img = attn_masks_img[:, :max_len_img]
        input_ids = input_ids[txt_indices, :max_len_txt]
        position_ids = position_ids[:, :max_len_txt]
        img_feat = img_feat[img_indices, :max_len_img]
        img_pos_feat = img_pos_feat[img_indices, :max_len_img]

        assert position_ids.size(0) == 1
        # if position_ids.size(0) != 1:
            # position_ids = position_ids[:self.hard_size+1]
        
        bs, max_tl = input_ids.size()
        lengths = txt_lens + num_bbs
        attention_mask = torch.arange(0, lengths.max()).type_as(lengths).repeat(bs, 1).lt(lengths.unsqueeze(1))
        out_size = attention_mask.size(1)
        gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size).to(input_ids.device)

        hard_batch['input_ids'] = input_ids
        hard_batch['position_ids'] = position_ids
        hard_batch['img_feat'] = img_feat
        hard_batch['img_pos_feat'] = img_pos_feat
        hard_batch['attn_masks'] = attention_mask
        hard_batch['attn_masks_txt'] = attn_masks_txt
        hard_batch['attn_masks_img'] = attn_masks_img
        hard_batch['gather_index'] = gather_index
        hard_batch['targets'] = targets
        return hard_batch

    

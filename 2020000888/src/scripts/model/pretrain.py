"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

UNITER for pretraining
"""
from collections import defaultdict
from pickle import NONE

import torch
from torch import nn
from torch.nn import functional as F
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

from .layer import GELU, BertOnlyMLMHead, BertOnlyCaptionHead, l2norm
from .model import UniterModel, UniterPreTrainedModel, UniterTwoFlowModel
from .transformer import DecoderLayer, Decoder
from .ot import optimal_transport_dist

class RegionFeatureRegression(nn.Module):
    " for MRM"
    def __init__(self, hidden_size, feat_dim, img_linear_weight):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 GELU(),
                                 LayerNorm(hidden_size, eps=1e-12))

        self.weight = img_linear_weight
        self.bias = nn.Parameter(torch.zeros(feat_dim))

    def forward(self, input_):
        hidden = self.net(input_)
        output = F.linear(hidden, self.weight.t(), self.bias)
        return output


class RegionClassification(nn.Module):
    " for MRC(-kl)"
    def __init__(self, hidden_size, label_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 GELU(),
                                 LayerNorm(hidden_size, eps=1e-12),
                                 nn.Linear(hidden_size, label_dim))

    def forward(self, input_):
        output = self.net(input_)
        return output


class UniterForPretraining(UniterPreTrainedModel):
    """ UNITER pretraining """
    def __init__(self, config, img_dim, img_label_dim):
        super().__init__(config)
        if config.model_mode == 'two_flow':
            print('Use the two flow model')
            self.two_flow=True
            self.uniter = UniterTwoFlowModel(config, img_dim)
        else:
            self.two_flow=False
            print('Use the one flow model')
            self.uniter = UniterModel(config, img_dim)
        # self.uniter.embeddings.word_embeddings.weight 是权重矩阵
        self.cls = BertOnlyMLMHead(
            config, self.uniter.embeddings.word_embeddings.weight)
        # anwen hu 2020/12/5
        self.cls_cap = BertOnlyCaptionHead(
            config, self.uniter.embeddings.word_embeddings.weight)
        self.feat_regress = RegionFeatureRegression(
            config.hidden_size, img_dim,
            self.uniter.img_embeddings.img_linear.weight)
        self.region_classifier = RegionClassification(
            config.hidden_size, img_label_dim)
        self.itm_output = nn.Linear(config.hidden_size, 2)
        self.ltm_output = nn.Linear(config.hidden_size, 2) # language text matching output 
        self.apply(self.init_weights)

    def forward(self, batch, task, compute_loss=True):
        '''
        input_ids torch.Size([8, 18]) = batch['input_ids'] 
        position_ids torch.Size([1, 18]) = batch['position_ids']
        img_feat torch.Size([8, 53, 2048]) = batch['img_feat']
        img_pos_feat torch.Size([8, 53, 7])  = batch['img_pos_feat']
        attention_mask torch.Size([8, 64]) = batch['attn_masks']
        attn_masks_txt = batch['attn_masks_txt']
        attn_masks_img = batch['attn_masks_img']
        gather_index torch.Size([8, 64]) = batch['gather_index']
        '''
        batch = defaultdict(lambda: None, batch)
        if task == 'mlm' or task == 'tlm':
            txt_labels = batch['txt_labels']
            return self.forward_mlm(batch, txt_labels, compute_loss)
        elif task == 'mrfr':
            img_mask_tgt = batch['img_mask_tgt']
            img_masks = batch['img_masks']
            mrfr_feat_target = batch['feat_targets']
            return self.forward_mrfr(batch, img_masks, img_mask_tgt,
                                     mrfr_feat_target, compute_loss)
        elif task == 'enitm':  # itm before the fusion encoder
            ot_inputs = batch['ot_inputs']
            return self.forward_en_itm(batch, ot_inputs, compute_loss)
        elif task == 'itm':
            targets = batch['targets']
            ot_inputs = batch['ot_inputs']
            return self.forward_itm(batch, targets, ot_inputs, compute_loss)
        elif task == 'itm_eval':
            assert compute_loss == False, 'doing full evaluation retrieval must not compute loss'
            scores, _ = self.forward_itm(batch, None, None, compute_loss)
            rank_scores = scores[:, 1:]
            return rank_scores
        elif task == 'cem':
            targets = batch['targets']
            return self.forward_cem(batch, targets, compute_loss)
        elif task == 'cemhn':
            return self.forward_cemhn(batch, compute_loss, hard_size=30)
        elif task == 'nceitm':
            return self.forward_nce_itm(batch, compute_loss)
        elif task.startswith('mrc'):
            img_mask_tgt = batch['img_mask_tgt']
            img_masks = batch['img_masks']
            mrc_label_target = batch['label_targets']
            return self.forward_mrc(batch, img_masks, img_mask_tgt,
                                    mrc_label_target, task, compute_loss)
        elif task.startswith('caption'):
            # caption task
            txt_labels = batch['txt_labels']
            return self.forward_caption(batch, txt_labels, compute_loss)
        else:
            raise ValueError('invalid task')

    def forward_mlm(self, batch, txt_labels, compute_loss=True):
        '''
        利用encoder最后一层的输出进行预测, 
        '''
        input_ids = batch['input_ids']
        # (batch, max-len, dim)
        if self.two_flow:  # anwen hu 2020/12/15 two flow model return three encoder output
            _, _, sequence_output = self.uniter(batch, output_all_encoded_layers=False)
        else:
            sequence_output = self.uniter(batch, output_all_encoded_layers=False)
        # get only the text part
        sequence_output = sequence_output[:, :input_ids.size(1), :]
        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    txt_labels != -1)
        prediction_scores = self.cls(masked_output)

        if compute_loss:
            masked_lm_loss = F.cross_entropy(prediction_scores,
                                             txt_labels[txt_labels != -1],
                                             reduction='none')
            return masked_lm_loss
        else:
            return prediction_scores

    def _compute_masked_hidden(self, hidden, mask):
        """ get only the masked region (don't compute unnecessary hiddens)
            mask = [[0, 0, 1, 0, 0]] 需要预测的位置为1，其他的不需要预测.
        """
        mask = mask.unsqueeze(-1).expand_as(hidden)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked

    def forward_mrfr(self, batch, img_masks, img_mask_tgt,
                     feat_targets, compute_loss=True):
        if self.two_flow:  # anwen hu 2020/12/15 two flow model return three encoder output
            _, _, sequence_output = self.uniter(batch, output_all_encoded_layers=False,img_masks=img_masks)
        else:
            sequence_output = self.uniter(batch, output_all_encoded_layers=False, img_masks=img_masks)

        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    img_mask_tgt)
        prediction_feat = self.feat_regress(masked_output)

        if compute_loss:
            mrfr_loss = F.mse_loss(prediction_feat, feat_targets,
                                   reduction='none')
            return mrfr_loss
        else:
            return prediction_feat

    def forward_itm(self, batch, targets, ot_inputs,
                    compute_loss=True):
        if self.two_flow: # anwen hu 2020/12/15 two flow model return three encoder output
            _, _, sequence_output = self.uniter(batch, output_all_encoded_layers=False)
        else:
            sequence_output = self.uniter(batch, output_all_encoded_layers=False)
        pooled_output = self.uniter.pooler(sequence_output)
        itm_scores = self.itm_output(pooled_output)

        # OT loss
        if ot_inputs is not None:
            input_ids = batch['input_ids']
            img_feat = batch['img_feat']
            ot_scatter = ot_inputs['ot_scatter']

            b = sequence_output.size(0)
            tl = input_ids.size(1)
            il = img_feat.size(1)
            max_l = max(ot_inputs['scatter_max'] + 1, tl+il)

            ot_scatter = ot_scatter.unsqueeze(-1).expand_as(sequence_output)
            ctx_emb = torch.zeros(b, max_l, self.config.hidden_size,
                                  dtype=sequence_output.dtype,
                                  device=sequence_output.device
                                  ).scatter_(dim=1, index=ot_scatter,
                                             src=sequence_output)
            txt_emb = ctx_emb[:, :tl, :]
            img_emb = ctx_emb[:, tl:tl+il, :]

            txt_pad = ot_inputs['txt_pad']
            img_pad = ot_inputs['img_pad']
            # NOTE: run in fp32 for stability
            ot_dist = optimal_transport_dist(txt_emb.float(), img_emb.float(),
                                             txt_pad, img_pad).to(txt_emb)
            ot_pos_dist = ot_dist.masked_select(targets == 1)
            ot_neg_dist = ot_dist.masked_select(targets == 0)
            ot_loss = (ot_pos_dist, ot_neg_dist)
        else:
            ot_loss = None

        if compute_loss:
            itm_loss = F.cross_entropy(itm_scores, targets, reduction='none')
            return itm_loss, ot_loss
        else:
            return itm_scores, ot_loss
    

    def forward_cem(self, batch, targets, compute_loss=True):
        sequence_output = self.uniter(batch, output_all_encoded_layers=False)
        pooled_output = self.uniter.pooler(sequence_output)
        ltm_scores = self.ltm_output(pooled_output)

        if compute_loss:
            ltm_loss = F.cross_entropy(ltm_scores, targets, reduction='none')
            return ltm_loss
        else:
            return ltm_scores

    def forward_cemhn(self, batch, compute_loss=True, hard_size=30):
        with torch.no_grad():
            self.eval()
            sequence_output = self.uniter(batch, output_all_encoded_layers=False)
            pooled_output = self.uniter.pooler(sequence_output)
            ltm_scores = self.ltm_output(pooled_output)
            rank_scores = ltm_scores[:, 1]
            hard_batch = cem_get_hard_batch(batch, rank_scores, hard_size)
        self.train()
        sequence_output = self.uniter(hard_batch, output_all_encoded_layers=False)
        pooled_output = self.uniter.pooler(sequence_output)
        ltm_scores = self.ltm_output(pooled_output)
        rank_scores = ltm_scores[:, 1]
        
        if compute_loss:
            ltm_loss = -F.log_softmax(rank_scores, dim=0)[:1]
            return ltm_loss
        else:
            return rank_scores

    def forward_en_itm(self, batch, ot_inputs, compute_loss=True):
        """
        anwen hu 2020/12/15:
        for en_item task, all pair in batch is positive pair, so there is no 'targets',
        negative pairs come from the batch
        :param batch:
        :param ot_inputs:
        :param compute_loss:
        :return:
        """
        assert self.two_flow
        assert ot_inputs is None
        ot_loss = None
        txt_encoded_output, img_encoded_output, sequence_output = self.uniter(batch, output_all_encoded_layers=False,
                                                                              fusion=False)
        # modified by zyd 2020.12.31
        txt_pooled_output = txt_encoded_output
        img_pooled_output = img_encoded_output
        # txt_pooled_output = torch.mean(txt_encoded_output, dim=1)  # batch * dim
        # img_pooled_output = torch.mean(img_encoded_output, dim=1)  # batch * dim

        # print('model/pretrain.py txt_pooled_output.shape', txt_pooled_output.shape)
        # print('model/pretrain.py img_pooled_output.shape', img_pooled_output.shape)
        # dot product: calculate similarity of each txt and img in the batch
        scores = torch.mm(txt_pooled_output, img_pooled_output.t())  # batch * batch

        if compute_loss:
            # contrastive loss: for each txt, use negative pair to estimate the probability of positive pair
            txt2img_loss = -F.log_softmax(scores / 0.07, dim=1).diag()  # batch ; retrieve image
            img2txt_loss = -F.log_softmax(scores / 0.07, dim=0).diag()  # batch ; retrieve txt
            en_itm_loss = txt2img_loss + img2txt_loss  # reduce the loss, enforce the pos_prob to 1
            return en_itm_loss, ot_loss
        else:
            en_itm_scores = scores  # batch * batch
            return en_itm_scores, ot_loss
    
    def forward_nce_itm(self, batch, compute_loss=True):
        """
        Liang Zhang 2021/1/13:
        for NCE itm task, the first pair of the batch is positive, and others are negative.
        :param batch:
        :param ot_inputs:
        :param compute_loss:
        :return:
        """

        batch_size = batch['input_ids'].size(0)
        loss = torch.zeros((batch_size), 
                            dtype=batch['img_feat'].dtype, device=batch['input_ids'].device)
        scores = torch.zeros((batch['input_ids'].size(0), batch['input_ids'].size(1)), 
                            dtype=batch['img_feat'].dtype, device=batch['input_ids'].device)
        for i in range(batch_size):
            batch_i = {}
            batch_i['input_ids'] = batch['input_ids'][i]
            batch_i['position_ids'] = batch['position_ids']
            batch_i['img_feat'] = batch['img_feat'][i]
            batch_i['img_pos_feat'] = batch['img_pos_feat'][i]
            batch_i['attn_masks'] = batch['attn_masks'][i]
            batch_i['attn_masks_txt'] = batch['attn_masks_txt'][i]
            batch_i['attn_masks_img'] = batch['attn_masks_img'][i]
            batch_i['gather_index'] = batch['gather_index'][i]
            #import pdb;pdb.set_trace()
        
            if self.two_flow:
                _, _, sequence_output = self.uniter(batch_i, output_all_encoded_layers=False)
            else:
                sequence_output = self.uniter(batch_i, output_all_encoded_layers=False)

            pooled_output = self.uniter.pooler(sequence_output)
            # rank_scores: (2*neg_size + 1)
            rank_scores = self.itm_output(pooled_output)[:, 1]
            scores[i, :] = rank_scores
            rank_loss = -F.log_softmax(rank_scores)[0]
            loss[i] = rank_loss

        if compute_loss:
            return loss
        else:
            return loss, scores



        if compute_loss:
            itm_loss = F.cross_entropy(itm_scores, targets, reduction='none')
            return itm_loss, ot_loss
        else:
            return itm_scores, ot_loss

                                                                    

    def forward_mrc(self, batch, img_masks, img_mask_tgt,
                    label_targets, task, compute_loss=True):
        if self.two_flow:  # anwen hu 2020/12/15 two flow model return three encoder output
            _, _, sequence_output = self.uniter(batch, output_all_encoded_layers=False, img_masks=img_masks)
        else:
            sequence_output = self.uniter(batch, output_all_encoded_layers=False, img_masks=img_masks)

        # only compute masked regions for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    img_mask_tgt)
        prediction_soft_label = self.region_classifier(masked_output)

        if compute_loss:
            if "kl" in task:
                prediction_soft_label = F.log_softmax(
                    prediction_soft_label, dim=-1)
                mrc_loss = F.kl_div(
                    prediction_soft_label, label_targets, reduction='none')
            else:
                # background class should not be the target
                label_targets = torch.max(label_targets[:, 1:], dim=-1)[1] + 1
                mrc_loss = F.cross_entropy(
                    prediction_soft_label, label_targets,
                    ignore_index=0, reduction='none')
            return mrc_loss
        else:
            return prediction_soft_label

    def forward_caption(self, batch, txt_labels, compute_loss=True):
        '''
        anwen hu:2020/11/22
        the only difference between caption and mlm is the text attention mask
        refer to "Unified Vision-Language Pre-Training for Image Captioning and VQA"
        '''
        input_ids = batch['input_ids']
        # (batch, max-len, dim)
        if self.two_flow: # anwen hu 2020/12/15 two flow model return three encoder output
            _, _, sequence_output = self.uniter(batch, output_all_encoded_layers=False)
        else:
            sequence_output = self.uniter(batch, output_all_encoded_layers=False)
        # get only the text part
        sequence_output = sequence_output[:, :input_ids.size(1), :]
        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    txt_labels != -1)
        # anwenhu 2020/12/5  revise cls to cls_cap
        prediction_scores = self.cls_cap(masked_output)

        if compute_loss:
            masked_lm_loss = F.cross_entropy(prediction_scores,
                                             txt_labels[txt_labels != -1],
                                             reduction='none')
            return masked_lm_loss
        else:
            return prediction_scores


# anwen hu 2020/12/9 add a decoder for caption
class UniterWithDecoderForPretraining(UniterPreTrainedModel):
    def __init__(self, config, img_dim, img_label_dim):
        super().__init__(config)
        if config.model_mode == 'two_flow':
            print('Use the two flow model')
            self.two_flow = True
            self.uniter = UniterTwoFlowModel(config, img_dim)
        else:
            print('Use the one flow model')
            self.two_flow = False
            self.uniter = UniterModel(config, img_dim)
        # self.uniter.embeddings.word_embeddings.weight 是权重矩阵
        self.cls = BertOnlyMLMHead(
            config, self.uniter.embeddings.word_embeddings.weight)
        # anwen hu 2020/12/8
        self.cls_cap = BertOnlyCaptionHead(
            config, self.uniter.embeddings.word_embeddings.weight)
        # pytorch >= 1.2.0
        # self.decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        # self.cap_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=config.num_decoder_layers)
        self.decoder_layer = DecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        print('model/pretrain.py: caption decoder layers ', config.num_decoder_layers)
        self.cap_decoder = Decoder(self.decoder_layer, num_layers=config.num_decoder_layers)
        self.feat_regress = RegionFeatureRegression(
            config.hidden_size, img_dim,
            self.uniter.img_embeddings.img_linear.weight)
        self.region_classifier = RegionClassification(
            config.hidden_size, img_label_dim)
        self.itm_output = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_weights)

    def forward(self, batch, task, compute_loss=True):
        '''
        input_ids torch.Size([8, 18]) = batch['input_ids']
        position_ids torch.Size([1, 18]) = batch['position_ids']
        img_feat torch.Size([8, 53, 2048]) = batch['img_feat']
        img_pos_feat torch.Size([8, 53, 7])  = batch['img_pos_feat']
        attention_mask torch.Size([8, 64]) = batch['attn_masks']
        attn_masks_txt = batch['attn_masks_txt']
        attn_masks_img = batch['attn_masks_img']
        gather_index torch.Size([8, 64]) = batch['gather_index']
        '''
        batch = defaultdict(lambda: None, batch)
        # print('model/pretrain.py forward ', task)
        if task == 'mlm':
            txt_labels = batch['txt_labels']
            return self.forward_mlm(batch, txt_labels, compute_loss)
        elif task == 'mrfr':
            img_mask_tgt = batch['img_mask_tgt']
            img_masks = batch['img_masks']
            mrfr_feat_target = batch['feat_targets']
            return self.forward_mrfr(batch, img_masks, img_mask_tgt,
                                     mrfr_feat_target, compute_loss)
        elif task == 'enitm':  # itm before the fusion encoder
            ot_inputs = batch['ot_inputs']
            return self.forward_en_itm(batch, ot_inputs, compute_loss)
        elif task == 'itm':
            targets = batch['targets']
            ot_inputs = batch['ot_inputs']
            return self.forward_itm(batch, targets, ot_inputs, compute_loss)
        elif task.startswith('mrc'):
            img_mask_tgt = batch['img_mask_tgt']
            img_masks = batch['img_masks']
            mrc_label_target = batch['label_targets']
            return self.forward_mrc(batch, img_masks, img_mask_tgt,
                                    mrc_label_target, task, compute_loss)
        elif task == 'caption-gen':
            # caption task
            txt_labels = batch['txt_labels']
            return self.forward_caption(batch, txt_labels, compute_loss)
        elif task == 'caption-autogen':
            # MM auto caption task
            txt_labels = batch['txt_labels']
            return self.forward_autocaption(batch, txt_labels, compute_loss)
        else:
            print('invalid task', task)
            raise ValueError('invalid task')

    def forward_mlm(self, batch, txt_labels, compute_loss=True):
        '''
        利用encoder最后一层的输出进行预测,
        '''
        input_ids = batch['input_ids']
        # (batch, max-len, dim)
        if self.two_flow:  # anwen hu 2020/12/15 two flow model return three encoder output
            _, _, sequence_output = self.uniter(batch, output_all_encoded_layers=False)
        else:
            sequence_output = self.uniter(batch, output_all_encoded_layers=False)
        # get only the text part
        sequence_output = sequence_output[:, :input_ids.size(1), :]
        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    txt_labels != -1)
        prediction_scores = self.cls(masked_output)

        if compute_loss:
            masked_lm_loss = F.cross_entropy(prediction_scores,
                                             txt_labels[txt_labels != -1],
                                             reduction='none')
            return masked_lm_loss
        else:
            return prediction_scores

    def _compute_masked_hidden(self, hidden, mask):
        """ get only the masked region (don't compute unnecessary hiddens)
            mask = [[0, 0, 1, 0, 0]] 需要预测的位置为1，其他的不需要预测.
        """
        mask = mask.unsqueeze(-1).expand_as(hidden)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked

    def forward_mrfr(self, batch, img_masks, img_mask_tgt,
                     feat_targets, compute_loss=True):
        if self.two_flow:  # anwen hu 2020/12/15 two flow model return three encoder output
            _, _, sequence_output = self.uniter(batch, output_all_encoded_layers=False, img_masks=img_masks)
        else:
            sequence_output = self.uniter(batch, output_all_encoded_layers=False, img_masks=img_masks)

        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    img_mask_tgt)
        prediction_feat = self.feat_regress(masked_output)

        if compute_loss:
            mrfr_loss = F.mse_loss(prediction_feat, feat_targets,
                                   reduction='none')
            return mrfr_loss
        else:
            return prediction_feat

    def forward_itm(self, batch, targets, ot_inputs,
                    compute_loss=True):
        if self.two_flow:  # anwen hu 2020/12/15 two flow model return three encoder output
            _, _, sequence_output = self.uniter(batch, output_all_encoded_layers=False)
        else:
            sequence_output = self.uniter(batch, output_all_encoded_layers=False)
        pooled_output = self.uniter.pooler(sequence_output)
        itm_scores = self.itm_output(pooled_output)

        # OT loss
        if ot_inputs is not None:
            input_ids = batch['input_ids']
            img_feat = batch['img_feat']
            ot_scatter = ot_inputs['ot_scatter']

            b = sequence_output.size(0)
            tl = input_ids.size(1)
            il = img_feat.size(1)
            max_l = max(ot_inputs['scatter_max'] + 1, tl + il)

            ot_scatter = ot_scatter.unsqueeze(-1).expand_as(sequence_output)
            ctx_emb = torch.zeros(b, max_l, self.config.hidden_size,
                                  dtype=sequence_output.dtype,
                                  device=sequence_output.device
                                  ).scatter_(dim=1, index=ot_scatter,
                                             src=sequence_output)
            txt_emb = ctx_emb[:, :tl, :]
            img_emb = ctx_emb[:, tl:tl + il, :]

            txt_pad = ot_inputs['txt_pad']
            img_pad = ot_inputs['img_pad']
            # NOTE: run in fp32 for stability
            ot_dist = optimal_transport_dist(txt_emb.float(), img_emb.float(),
                                             txt_pad, img_pad).to(txt_emb)
            ot_pos_dist = ot_dist.masked_select(targets == 1)
            ot_neg_dist = ot_dist.masked_select(targets == 0)
            ot_loss = (ot_pos_dist, ot_neg_dist)
        else:
            ot_loss = None

        if compute_loss:
            itm_loss = F.cross_entropy(itm_scores, targets, reduction='none')
            return itm_loss, ot_loss
        else:
            return itm_scores, ot_loss

    def forward_en_itm(self, batch, ot_inputs, compute_loss=True):
        """
        anwen hu 2020/12/15:
        for en_item task, all pair in batch is positive pair, so there is no 'targets',
        negative pairs come from the batch
        :param batch:
        :param ot_inputs:
        :param compute_loss:
        :return:
        """
        assert self.two_flow
        assert ot_inputs is None
        ot_loss = None
        txt_encoded_output, img_encoded_output, sequence_output = self.uniter(batch, output_all_encoded_layers=False,
                                                                              fusion=False)
        # modified by zyd 2020.12.31
        txt_pooled_output = txt_encoded_output
        img_pooled_output = img_encoded_output
        # txt_pooled_output = torch.mean(txt_encoded_output, dim=1)  # batch * dim
        # img_pooled_output = torch.mean(img_encoded_output, dim=1)  # batch * dim
        
        # print('model/pretrain.py txt_pooled_output.shape', txt_pooled_output.shape)
        # print('model/pretrain.py img_pooled_output.shape', img_pooled_output.shape)
        # dot product: calculate similarity of each txt and img in the batch
        scores = torch.mm(txt_pooled_output, img_pooled_output.t())  # batch * batch

        if compute_loss:
            # contrastive loss: for each txt, use negative pair to estimate the probability of positive pair
            txt2img_loss = -F.log_softmax(scores / 0.07, dim=1).diag()  # batch ; retrieve image
            img2txt_loss = -F.log_softmax(scores / 0.07, dim=0).diag()  # batch ; retrieve txt
            en_itm_loss = txt2img_loss + img2txt_loss  # reduce the loss, enforce the pos_prob to 1
            return en_itm_loss, ot_loss
        else:
            en_itm_scores = scores  # batch * batch
            return en_itm_scores, ot_loss

    def forward_mrc(self, batch, img_masks, img_mask_tgt,
                    label_targets, task, compute_loss=True):
        if self.two_flow:  # anwen hu 2020/12/15 two flow model return three encoder output
            _, _, sequence_output = self.uniter(batch, output_all_encoded_layers=False, img_masks=img_masks)
        else:
            sequence_output = self.uniter(batch, output_all_encoded_layers=False, img_masks=img_masks)

        # only compute masked regions for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    img_mask_tgt)
        prediction_soft_label = self.region_classifier(masked_output)

        if compute_loss:
            if "kl" in task:
                prediction_soft_label = F.log_softmax(
                    prediction_soft_label, dim=-1)
                mrc_loss = F.kl_div(
                    prediction_soft_label, label_targets, reduction='none')
            else:
                # background class should not be the target
                label_targets = torch.max(label_targets[:, 1:], dim=-1)[1] + 1
                mrc_loss = F.cross_entropy(
                    prediction_soft_label, label_targets,
                    ignore_index=0, reduction='none')
            return mrc_loss
        else:
            return prediction_soft_label

    def forward_caption(self, batch, txt_labels, compute_loss=True):
        '''
        anwen hu:2020/12/8
        an additional decoder for image captioning
        '''
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        txt_emb = self.uniter._compute_txt_embeddings(input_ids, position_ids) # batch * max_txt_num * dim
        batch['input_ids'] = None
        # print('model/pretrain.py attn_masks_img', batch['attn_masks_img'].shape)
        batch['attn_masks'] = batch['attn_masks_img']
        assert len(batch['attn_masks'].shape) == 2
        if self.two_flow: # anwen hu 2020/12/15 for two flow model, only fetch the image encoder output
            _, img_memory, _ = self.uniter(batch, output_all_encoded_layers=False)  # batch * max_img_num * dim
        else:
            img_memory = self.uniter(batch, output_all_encoded_layers=False) # batch * max_img_num * dim
        #  anwen hu 2020/12/9 for torch.nn.TransformerDecoder
        """txt_emb = txt_emb.transpose(0, 1)  # max_txt_num * batch *dim
        img_memory = img_memory.transpose(0, 1)  # max_img_num * batch * dim
        memory_mask = torch.ones([img_memory.size(0), img_memory.size(0)], dtype=torch.long)
        tgt_mask=batch['attn_masks_txt'][0]"""
        # print('model/pretrain.py txt_emb:', txt_emb.shape)
        # print('model/pretrain.py img_memory:', img_memory.shape)
        # print('model/pretrain.py tgt_mask[0]:', batch['attn_masks_txt'][0])
        # assert torch.equal(batch['attn_masks_txt'][0], batch['attn_masks_txt'][4])
        memory_mask = batch['attn_masks_img']
        tgt_mask = batch['attn_masks_txt']
        # print('model/pretrain.py memory_mask:', memory_mask)
        sequence_output = self.cap_decoder(tgt=txt_emb,  # batch * max_txt_num * dim
                                           memory=img_memory,  # batch * max_img_num * dim
                                           tgt_mask=tgt_mask,  # batch * max_txt_num * max_txt_num
                                           memory_mask=memory_mask)  # batch * max_img_num

        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    txt_labels != -1)
        # anwenhu 2020/12/5  revise cls to cls_cap
        prediction_scores = self.cls_cap(masked_output)

        if compute_loss:
            masked_lm_loss = F.cross_entropy(prediction_scores,
                                             txt_labels[txt_labels != -1],
                                             reduction='none')
            return masked_lm_loss
        else:
            return prediction_scores

    def forward_autocaption(self, batch, txt_labels, compute_loss=True):
        '''
        anwenhu 2020/12/23
        feed masked caption in multi-modal encoder, use fused features to generate the whole sentence
        '''
        # print('model/pretrain.py dec_input_ids[:2]:', batch['dec_input_ids'][:2])
        # print('model/pretrain.py input_ids[:2]:', batch['input_ids'][:2])
        dec_input_ids = batch['dec_input_ids']
        position_ids = batch['position_ids']
        # used for decoder
        txt_emb = self.uniter._compute_txt_embeddings(dec_input_ids, position_ids)  # batch * max_txt_num * dim
        # print('model/pretrain.py attn_masks_img', batch['attn_masks_img'].shape)
        assert len(batch['attn_masks'].shape) == 2
        if self.two_flow:  # anwen hu 2020/12/23 for two flow model, fetch the fusion encoder output
            _, _, fusion_memory = self.uniter(batch, output_all_encoded_layers=False)  # batch * max_img_txt_num * dim
        else:
            fusion_memory = self.uniter(batch, output_all_encoded_layers=False)  # batch * max_img_txt_num * dim
        # print('model/pretrain.py txt_emb:', txt_emb.shape)
        # print('model/pretrain.py fusion_memory:', fusion_memory.shape)
        # print('model/pretrain.py tgt_mask[0]:', batch['dec_attn_masks_txt'][0])
        # assert torch.equal(batch['dec_attn_masks_txt'][0], batch['dec_attn_masks_txt'][4])
        memory_mask = batch['attn_masks']
        tgt_mask = batch['dec_attn_masks_txt']
        # print('model/pretrain.py memory_mask:', memory_mask)
        sequence_output = self.cap_decoder(tgt=txt_emb,  # batch * max_txt_num * dim
                                           memory=fusion_memory,  # batch * max_img_txt_num * dim
                                           tgt_mask=tgt_mask,  # batch * max_txt_num * max_txt_num
                                           memory_mask=memory_mask)  # batch * max_img_txt_num

        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    txt_labels != -1)
        prediction_scores = self.cls_cap(masked_output)

        if compute_loss:
            masked_lm_loss = F.cross_entropy(prediction_scores,
                                             txt_labels[txt_labels != -1],
                                             reduction='none')
            return masked_lm_loss
        else:
            return prediction_scores


def cem_get_hard_batch(batch, scores, hard_size):
    batch = defaultdict(lambda: None, batch)
    input_ids = batch['input_ids']
    position_ids = batch['position_ids']
    img_feat = batch['img_feat']
    img_pos_feat = batch['img_pos_feat']
    attention_mask = batch['attn_masks']
    attn_masks_txt = batch['attn_masks_txt']
    attn_masks_img = batch['attn_masks_img']
    gather_index = batch['gather_index']
    hard_batch = {'sample_size': hard_size + 1}

    # NOTE first example is positive
    hard_indices = scores.squeeze(-1)[1:].topk(
        hard_size, sorted=False)[1] + 1
    indices = torch.cat([torch.zeros(1, dtype=torch.long,
                                        device=hard_indices.device),
                            hard_indices])

    attention_mask = attention_mask.index_select(0, indices)
    gather_index = gather_index.index_select(0, indices)
    
    # add by zyd 2020.12.28
    input_ids = input_ids.index_select(0, indices)
    attn_masks_txt = attn_masks_txt.index_select(0, indices)

    position_ids = position_ids.index_select(0, indices)
    if img_feat is not None:
        img_feat = img_feat.index_select(0, indices)
        img_pos_feat = img_pos_feat.index_select(0, indices)
        attn_masks_img = attn_masks_img.index_select(0, indices)

    hard_batch['input_ids'] = input_ids
    hard_batch['position_ids'] = position_ids
    hard_batch['img_feat'] = img_feat
    hard_batch['img_pos_feat'] = img_pos_feat
    hard_batch['attn_masks'] = attention_mask
    hard_batch['attn_masks_txt'] = attn_masks_txt
    hard_batch['attn_masks_img'] = attn_masks_img
    hard_batch['gather_index'] = gather_index

    return hard_batch
"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

UNITER for Image Captioning model: anwen hu 2020/11/24
"""
from collections import defaultdict
from torch.nn import functional as F
import torch
from torch import nn
from .model import UniterPreTrainedModel, UniterModel, UniterTwoFlowModel
from .layer import GELU, BertOnlyMLMHead, BertOnlyCaptionHead
from .transformer import Decoder, DecoderLayer


class UniterForImageCaptioning(UniterPreTrainedModel):
    """ Finetune UNITER for image captioning
    """

    def __init__(self, config, img_dim):
        super().__init__(config)
        # self.uniter = UniterModel(config, img_dim)
        # anwen hu 2020/12/20: use one-flow or two-flow model
        if config.model_mode == 'two_flow':
            print('Use the two flow model')
            self.two_flow = True
            self.uniter = UniterTwoFlowModel(config, img_dim)
        else:
            self.two_flow = False
            print('Use the one flow model')
            self.uniter = UniterModel(config, img_dim)
        # self.cls = BertOnlyMLMHead(config, self.uniter.embeddings.word_embeddings.weight)
        self.cls_cap = BertOnlyCaptionHead(config, self.uniter.embeddings.word_embeddings.weight)
        self.apply(self.init_weights)

    def _compute_masked_hidden(self, hidden, mask):
        # anwen hu 2020/11/30: copy from pretraining class,
        """ get only the masked region (don't compute unnecessary hiddens)
            mask = [[1, 1, 1, 1,..., -1]] 
        """
        mask = mask.unsqueeze(-1).expand_as(hidden)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked

    def forward(self, batch, compute_loss=True):
        # print('model/caption.py batch', type(batch))
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        txt_labels = batch['txt_labels']
        # print('model/caption.py input_ids', input_ids)
        # print('model/caption.py txt_labels', txt_labels)
        if self.two_flow:
            _, _, sequence_output = self.uniter(batch, output_all_encoded_layers=False)
        else:
            sequence_output = self.uniter(batch, output_all_encoded_layers=False)
        # get only the text part
        sequence_output = sequence_output[:, :input_ids.size(1), :]
        # only compute masked tokens for better efficiency during training
        if compute_loss:
            masked_output = self._compute_masked_hidden(sequence_output,
                                                        txt_labels != -1)  # for fine-tune, only cls and sep is -1
            prediction_scores = self.cls_cap(masked_output)
        else:
            # print('model/caption.py sequence_output:', sequence_output.shape) # batch * max_decoding_len * D_h
            flat_prediction_scores = self.cls_cap(sequence_output.contiguous().view(-1, sequence_output.size(-1)))
            # print('model/caption.py flat prediction_scores:', flat_prediction_scores.shape)
            # (batch*max_decoding_len) * vocab_size
            prediction_scores = flat_prediction_scores.contiguous().view(input_ids.size(0), input_ids.size(1), -1)

        if compute_loss:
            masked_lm_loss = F.cross_entropy(prediction_scores,
                                             txt_labels[txt_labels != -1],
                                             reduction='none')
            return masked_lm_loss
        else:
            return prediction_scores


# anwen hu 2020/12/8 add an additional decoder
class UniterWithDecoderForImageCaptioning(UniterPreTrainedModel):
    """ Finetune UNITER with decoder for image captioning
    """

    def __init__(self, config, img_dim, freeze_encoder):
        super().__init__(config)
        # anwen hu 2020/12/11 whether to freeze encoder when during fine-tune
        self.freeze_encoder = freeze_encoder
        # self.uniter = UniterModel(config, img_dim)
        # anwen hu 2020/12/20: use one-flow or two-flow model
        if config.model_mode == 'two_flow':
            print('Use the two flow model')
            self.two_flow = True
            self.uniter = UniterTwoFlowModel(config, img_dim)
        else:
            self.two_flow = False
            print('Use the one flow model')
            self.uniter = UniterModel(config, img_dim)
        self.cls_cap = BertOnlyCaptionHead(config, self.uniter.embeddings.word_embeddings.weight)
        # anwen hu 2020/12/9 pytorch >= 1.2.0
        # self.decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        # self.cap_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=config.num_decoder_layers)
        self.decoder_layer = DecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        self.cap_decoder = Decoder(self.decoder_layer, num_layers=config.num_decoder_layers)
        self.apply(self.init_weights)

    def _compute_masked_hidden(self, hidden, mask):
        # anwen hu 2020/11/30: copy from pretraining class,
        """ get only the masked region (don't compute unnecessary hiddens)
            mask = [[1, 1, 1, 1,..., -1]]
        """
        mask = mask.unsqueeze(-1).expand_as(hidden)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked

    def forward(self, batch, compute_loss=True):
        '''
        anwen hu:2020/12/8
        an additional decoder for image captioning
        '''
        input_ids = batch['input_ids']
        txt_labels = batch['txt_labels']
        position_ids = batch['position_ids']
        batch['input_ids'] = None
        batch['attn_masks'] = batch['attn_masks_img']
        if self.freeze_encoder:
            with torch.no_grad():
                if self.two_flow:
                    _, img_memory, _ = self.uniter(batch, output_all_encoded_layers=False)  # batch * max_img_num * dim
                else:
                    img_memory = self.uniter(batch, output_all_encoded_layers=False)  # batch * max_img_num * dim
                txt_emb = self.uniter._compute_txt_embeddings(input_ids, position_ids)
        else:
            if self.two_flow:
                _, img_memory, _ = self.uniter(batch, output_all_encoded_layers=False)  # batch * max_img_num * dim
            else:
                img_memory = self.uniter(batch, output_all_encoded_layers=False)  # batch * max_img_num * dim
            txt_emb = self.uniter._compute_txt_embeddings(input_ids, position_ids)
        batch['input_ids'] = None
        # anwen hu 2020/12/11 only designed for inference, restore input_ids
        batch['input_ids'] = input_ids
        #  anwen hu 2020/12/9 for torch.nn.TransformerDecoder
        """txt_emb = txt_emb.transpose(0, 1)  # max_txt_num * batch *dim
        img_memory = img_memory.transpose(0, 1)  # max_img_num * batch * dim
        memory_mask = torch.ones([img_memory.size(0), img_memory.size(0)], dtype=torch.long)
        tgt_mask=batch['attn_masks_txt'][0]"""
        # print('model/caption.py txt_emb:', txt_emb.shape)
        # print('model/caption.py img_memory:', img_memory.shape)
        # print('model/caption.py tgt_mask[0]:', batch['attn_masks_txt'][0])
        # assert torch.equal(batch['attn_masks_txt'][0], batch['attn_masks_txt'][4])
        memory_mask = batch['attn_masks_img']
        tgt_mask = batch['attn_masks_txt']
        # print('model/caption.py memory_mask:', memory_mask)
        sequence_output = self.cap_decoder(tgt=txt_emb,  # batch * max_txt_num * dim
                                           memory=img_memory,  # batch * max_img_num * dim
                                           tgt_mask=tgt_mask,  # batch * max_txt_num * max_txt_num
                                           memory_mask=memory_mask)  # batch * max_img_num

        # print('model/caption.py sequence_output: ', sequence_output.shape)
        if compute_loss:
            # only compute masked tokens for better efficiency
            masked_output = self._compute_masked_hidden(sequence_output, txt_labels != -1)
            prediction_scores = self.cls_cap(masked_output)

        else:
            # print('model/caption.py sequence_output:', sequence_output.shape) # batch * max_decoding_len * D_h
            flat_prediction_scores = self.cls_cap(sequence_output.contiguous().view(-1, sequence_output.size(-1)))
            # print('model/caption.py flat prediction_scores:', flat_prediction_scores.shape)
            # (batch*max_decoding_len) * vocab_size
            prediction_scores = flat_prediction_scores.contiguous().view(input_ids.size(0), input_ids.size(1), -1)

        if compute_loss:
            masked_lm_loss = F.cross_entropy(prediction_scores,
                                             txt_labels[txt_labels != -1],
                                             reduction='none')
            return masked_lm_loss
        else:
            return prediction_scores

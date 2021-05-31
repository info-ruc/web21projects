"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Pytorch modules
some classes are modified from HuggingFace
(https://github.com/huggingface/transformers)
"""
import copy
import json
import logging
from io import open
from tempfile import tempdir

import torch
from torch import nn
from apex.normalization.fused_layer_norm import FusedLayerNorm

from .layer import BertLayer, BertPooler, BertTwoFlowPooler

logger = logging.getLogger(__name__)


class UniterConfig(object):
    """Configuration class to store the configuration of a `UniterModel`.
    """

    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        """Constructs UniterConfig.
        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in
                `UniterModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer
                encoder.
            num_attention_heads: Number of attention heads for each attention
                layer in the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e.
                feed-forward) layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string)
                in the encoder and pooler. If string, "gelu", "relu" and
                "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully
                connected layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this
                model might ever be used with. Typically set this to something
                large just in case (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed
                into `UniterModel`.
            initializer_range: The sttdev of the truncated_normal_initializer
                for initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file,
                      "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size "
                             "(int) or the path to a pretrained model config "
                             "file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `UniterConfig` from a
           Python dictionary of parameters."""
        config = UniterConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `UniterConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class UniterPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, UniterConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of "
                "class `UniterConfig`. To create a model from a Google "
                "pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses
            # truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
        elif isinstance(module, FusedLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, config_file, state_dict, *inputs, **kwargs):
        """
        Instantiate a UniterPreTrainedModel from a pre-trained model file or a
        pytorch state dict.
        Params:
            config_file: config json file
            state_dict: an state dictionnary
            *inputs, **kwargs: additional input for the specific Uniter class
        """
        # Load config
        config = UniterConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = ({} if metadata is None
                              else metadata.get(prefix[:-1], {}))
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys,
                unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.')
                                              for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from "
                        "pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in "
                        "{}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for '
                               '{}:\n\t{}'.format(
                model.__class__.__name__,
                "\n\t".join(error_msgs)))
        return model


class UniterTextEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        '''
        获取最后的文本的embeddings, 作为TextEncoder 或者 统一的Encoder 的输入
        embeddings = tokenEmd + PosEmd + tokentypeEmb
        如果采用双流的模型的话，那么这里type embedding可以去掉。
        '''
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        if config.model_mode != 'two_flow':
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                                      config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model
        # variable name and be able to load any TensorFlow checkpoint file
        self.LayerNorm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, position_ids, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids).long()
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        if self.config.model_mode == 'two_flow':
            embeddings = (words_embeddings
                          + position_embeddings)
        else:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings = (words_embeddings
                          + position_embeddings
                          + token_type_embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class UniterImageEmbeddings(nn.Module):
    def __init__(self, config, img_dim):
        super().__init__()
        '''
        获取最后的图像的embeddings, 作为 ImgEncoder 或者 统一的 Encoder 的输入
        embeddings = imageEmd + PosEmd + typeEmb
        '''
        self.config = config
        self.img_linear = nn.Linear(img_dim, config.hidden_size)
        self.img_layer_norm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.pos_layer_norm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.pos_linear = nn.Linear(7, config.hidden_size)
        self.mask_embedding = nn.Embedding(2, img_dim, padding_idx=0)

        # tf naming convention for layer norm
        self.LayerNorm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, img_feat, img_pos_feat, type_embeddings, img_masks=None):
        if img_masks is not None:
            self.mask_embedding.weight.data[0, :].fill_(0)
            mask = self.mask_embedding(img_masks.long())
            img_feat = img_feat + mask

        transformed_im = self.img_layer_norm(self.img_linear(img_feat))
        transformed_pos = self.pos_layer_norm(self.pos_linear(img_pos_feat))
        if self.config.model_mode == 'two_flow':
            embeddings = transformed_im + transformed_pos
        else:
            embeddings = transformed_im + transformed_pos + type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class UniterEncoder(nn.Module):
    def __init__(self, config, num_layers=None):
        # 在双流模型中，需要指定num_layers建立不同的encoder, textEncoder, ImgEncoder
        # 在单流模型中，那么采用配置文件中 num_hidden_layers
        super().__init__()
        layer = BertLayer(config)
        if num_layers is None:
            num_layers = config.num_hidden_layers
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])

    def forward(self, input_, attention_mask, frozen_en_layer=0,
                output_all_encoded_layers=True):
        # anwen hu 2020/12/15: add parameter: frozen_en_layer
        all_encoder_layers = []
        hidden_states = input_
        for i, layer_module in enumerate(self.layer):
            if i <= frozen_en_layer - 1:
                with torch.no_grad():
                    hidden_states = layer_module(hidden_states, attention_mask)
            else:
                hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class UniterModel(UniterPreTrainedModel):
    """ Modification for Joint Vision-Language Encoding
    """

    def __init__(self, config, img_dim):
        super().__init__(config)
        self.embeddings = UniterTextEmbeddings(config)
        self.img_embeddings = UniterImageEmbeddings(config, img_dim)
        self.encoder = UniterEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_weights)

    def _compute_txt_embeddings(self, input_ids, position_ids,
                                txt_type_ids=None):
        output = self.embeddings(input_ids, position_ids, txt_type_ids)
        return output

    def _compute_img_embeddings(self, img_feat, img_pos_feat, img_masks=None,
                                img_type_ids=None):
        '''img_type_ids: 目前pretrain.py中传递的都是None, 这里应该是预留的接口，暂时没有用到.
        '''
        if img_type_ids is None:
            img_type_ids = torch.ones_like(img_feat[:, :, 0].long())
        img_type_embeddings = self.embeddings.token_type_embeddings(
            img_type_ids)
        output = self.img_embeddings(img_feat, img_pos_feat,
                                     img_type_embeddings, img_masks)
        return output

    def _compute_img_txt_embeddings(self, input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    gather_index, img_masks=None,
                                    txt_type_ids=None, img_type_ids=None):
        txt_emb = self._compute_txt_embeddings(
            input_ids, position_ids, txt_type_ids)
        img_emb = self._compute_img_embeddings(
            img_feat, img_pos_feat, img_masks, img_type_ids)
        # align back to most compact input
        gather_index = gather_index.unsqueeze(-1).expand(
            -1, -1, self.config.hidden_size)
        embedding_output = torch.gather(torch.cat([txt_emb, img_emb], dim=1),
                                        dim=1, index=gather_index)
        return embedding_output

    def forward(self, batch, img_masks=None,
                output_all_encoded_layers=True,
                txt_type_ids=None, img_type_ids=None):
        '''
        :input_ids torch.Size([8, 18])
        :position_ids torch.Size([1, 18])
        :img_feat torch.Size([8, 53, 2048])
        :img_pos_feat torch.Size([8, 53, 7])
        :attention_mask torch.Size([8, 64])
        :gather_index torch.Size([8, 64])
        :encoded_layers torch.Size([8, 64, 768])
        '''
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attention_mask = batch['attn_masks']
        gather_index = batch['gather_index']
        # compute self-attention mask
        #  anwen hu 2020/11/22: for captioning task, dim(attention_mask)=3
        if len(attention_mask.shape) == 3:  # only designed for captioning task in unified encoder-decoder
            extended_attention_mask = attention_mask.unsqueeze(
                1)  # batch * max(L+bb_num) * max(L+bb_num) > batch * 1 * max(L+bb_num) * max(L+bb_num)
        else:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(
                2)  # batch * max(L+bb_num) > batch * 1 * 1 * max(L+bb_num)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # embedding layer
        if input_ids is None:
            # image only
            embedding_output = self._compute_img_embeddings(
                img_feat, img_pos_feat, img_masks, img_type_ids)
        elif img_feat is None:
            # text only
            embedding_output = self._compute_txt_embeddings(
                input_ids, position_ids, txt_type_ids)
        else:
            # print('model/model/py uniter compute_img_txt_embeddings')
            embedding_output = self._compute_img_txt_embeddings(
                input_ids, position_ids,
                img_feat, img_pos_feat,
                gather_index, img_masks, txt_type_ids, img_type_ids)

        encoded_layers = self.encoder(
            embedding_output, extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers


class UniterTwoFlowModel(UniterPreTrainedModel):
    """ Modification for Joint Vision-Language Encoding
        Anwen Hu 2020/12/10
        Tow separate transformer encoder for txt and image
        One multimodal encoder to fuse txt and image output
    """

    def __init__(self, config, img_dim):
        super().__init__(config)
        assert config.model_mode == 'two_flow'
        self.frozen_txt_encoder_layer = config.frozen_txt_encoder_layer
        # embeddings = tokenEmd + PosiEmd, 不加 typeEmb
        self.embeddings = UniterTextEmbeddings(config)
        # img_embeddings = RegionEmd + PosEmd, 不加 typeEmb
        self.img_embeddings = UniterImageEmbeddings(config, img_dim)
        # anwen hu 2020/12/15
        # txt encoder is named as self.encoder to be consistent with the init uniter model(come from txt bert model)
        self.encoder = UniterEncoder(config, config.txt_encoder_layers)
        self.encoder_img = UniterEncoder(config, config.img_encoder_layers)
        self.encoder_fusion = UniterEncoder(config, config.fusion_encoder_layers)
        self.pooler = BertPooler(config)
        # add by zyd 2020.12.31
        self.img_pooler = BertTwoFlowPooler(config)
        self.txt_pooler = BertTwoFlowPooler(config)

        self.apply(self.init_weights)

    def _compute_txt_embeddings(self, input_ids, position_ids,
                                txt_type_ids=None):
        output = self.embeddings(input_ids, position_ids, txt_type_ids)
        return output

    def _compute_img_embeddings(self, img_feat, img_pos_feat, img_masks=None,
                                img_type_ids=None):
        img_type_embeddings = None
        output = self.img_embeddings(img_feat, img_pos_feat,
                                     img_type_embeddings, img_masks)
        return output

    def gather_embeddings(self, txt_emb, img_emb, gather_index):
        gather_index = gather_index.unsqueeze(-1).expand(-1, -1, self.config.hidden_size)
        embedding_output = torch.gather(torch.cat([txt_emb, img_emb], dim=1),
                                        dim=1, index=gather_index)
        return embedding_output

    def forward(self, batch, img_masks=None,
                output_all_encoded_layers=True,
                txt_type_ids=None, img_type_ids=None, fusion=True, fix_encoder=False, encoder_out=False):
        '''
        output: txt_encoded_layers, img_encoded_layers, fusion_encoded_layers
        batch 内存放的通用的
        '''
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attention_mask = batch['attn_masks']
        attention_mask_txt = batch['attn_masks_txt']
        attention_mask_img = batch['attn_masks_img']
        gather_index = batch['gather_index']
        # compute self-attention mask
        if len(attention_mask.shape) == 3:  # only designed for captioning task in unified encoder-decoder
            # batch * max(L+bb_num) * max(L+bb_num) > batch * 1 * max(L+bb_num) * max(L+bb_num)
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            # batch * max(L+bb_num) > batch * 1 * 1 * max(L+bb_num)
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        # mask 为 0的部分, 赋值 -10000.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # txt_mask
        if input_ids is not None:
            if len(attention_mask.shape) == 3:  # only designed for captioning task in unified encoder-decoder
                # batch * max(L) * max(L) > batch * 1 * max(L) * max(L)
                extended_attention_mask_txt = attention_mask_txt.unsqueeze(1)
            else:
                # batch * max(L) > batch * 1 * 1 * max(L)
                extended_attention_mask_txt = attention_mask_txt.unsqueeze(1).unsqueeze(2)
            extended_attention_mask_txt = extended_attention_mask_txt.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask_txt = (1.0 - extended_attention_mask_txt) * -10000.0
        # img_mask
        if img_feat is not None:
            extended_attention_mask_img = attention_mask_img.unsqueeze(1).unsqueeze(2)
            extended_attention_mask_img = extended_attention_mask_img.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask_img = (1.0 - extended_attention_mask_img) * -10000.0

        if input_ids is None:
            # image only
            img_embd = self._compute_img_embeddings(img_feat, img_pos_feat, img_masks, img_type_ids)
            img_encoded_layers = self.encoder_img(img_embd, extended_attention_mask_img,
                                                  output_all_encoded_layers=output_all_encoded_layers)
            if not output_all_encoded_layers:
                img_encoded_layers = img_encoded_layers[-1]
            txt_encoded_layers = None
            fusion_encoded_layers = None
            # anwen hu 2020/12/10: only image, don't need to use encoder_fusion

            # add by zyd 2020.12.31
            img_encoded_layers = self.img_pooler(img_encoded_layers, attention_mask_img)

            return txt_encoded_layers, img_encoded_layers, fusion_encoded_layers
        elif img_feat is None:
            # text only
            txt_embd = self._compute_txt_embeddings(input_ids, position_ids, txt_type_ids)
            txt_encoded_layers = self.encoder(txt_embd, extended_attention_mask_txt,
                                              frozen_en_layer=self.frozen_txt_encoder_layer,
                                              output_all_encoded_layers=output_all_encoded_layers)
            if not output_all_encoded_layers:
                txt_encoded_layers = txt_encoded_layers[-1]
            # anwen hu 2020/12/10: only txt, don't need to use encoder_fusion
            img_encoded_layers = None
            fusion_encoded_layers = None

            # add by zyd 2020.12.31
            txt_encoded_layers = self.txt_pooler(txt_encoded_layers, attention_mask_txt)
            
            return txt_encoded_layers, img_encoded_layers, fusion_encoded_layers

        else:
            img_embd = self._compute_img_embeddings(img_feat, img_pos_feat, img_masks, img_type_ids)
            img_encoded_layers = self.encoder_img(img_embd, extended_attention_mask_img,
                                                  output_all_encoded_layers=output_all_encoded_layers)
            # print('model/model.py img_encoded_layers[-1]:', img_encoded_layers[-1].shape)
            txt_embd = self._compute_txt_embeddings(input_ids, position_ids, txt_type_ids)
            txt_encoded_layers = self.encoder(txt_embd, extended_attention_mask_txt,
                                              frozen_en_layer=self.frozen_txt_encoder_layer,
                                              output_all_encoded_layers=output_all_encoded_layers)
            # print('model/model.py txt_encoded_layers[-1]:', txt_encoded_layers[-1].shape)
            if fusion:  # whether to fuse txt and img feature in fusion encoders
                txt_img_embedding = self.gather_embeddings(txt_emb=txt_encoded_layers[-1], img_emb=img_encoded_layers[-1],
                                                           gather_index=gather_index)
                #import pdb;pdb.set_trace()
                if fix_encoder:
                    txt_img_embedding = txt_img_embedding.detach()
                fusion_encoded_layers = self.encoder_fusion(
                    txt_img_embedding, extended_attention_mask,
                    output_all_encoded_layers=output_all_encoded_layers)
                if not output_all_encoded_layers:
                    txt_encoded_layers = txt_encoded_layers[-1]
                    img_encoded_layers = img_encoded_layers[-1]
                    fusion_encoded_layers = fusion_encoded_layers[-1]
            else:
                if not output_all_encoded_layers:
                    txt_encoded_layers = txt_encoded_layers[-1]
                    img_encoded_layers = img_encoded_layers[-1]
                    fusion_encoded_layers = None
            
            # add by zyd 2020.12.31
            txt_pooled_output = self.txt_pooler(txt_encoded_layers, attention_mask_txt)
            img_pooled_output = self.img_pooler(img_encoded_layers, attention_mask_img)
            
            # add by zhangliang 2021/1/14
            if encoder_out:
                return txt_pooled_output, img_pooled_output, txt_encoded_layers, img_encoded_layers

            return txt_pooled_output, img_pooled_output, fusion_encoded_layers
    
    # add by zhangliang 2012/1/14
    def forward_fusion(self, batch):
        with torch.no_grad():
            self.eval()
            txt_pooled_output, img_pooled_output, txt_encoded_layers, img_encoded_layers = self.forward(batch, output_all_encoded_layers=False, fix_encoder=True, encoder_out=True)
            self.train()
        txt_emb = batch['txt_emb']
        img_emb = batch['img_emb']
        gather_index = batch['gather_index']
        attention_mask = batch['attn_masks']
        assert txt_emb is not None
        assert img_emb is not None
        # compute self-attention mask
        if len(attention_mask.shape) == 3:  # only designed for captioning task in unified encoder-decoder
            # batch * max(L+bb_num) * max(L+bb_num) > batch * 1 * max(L+bb_num) * max(L+bb_num)
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            # batch * max(L+bb_num) > batch * 1 * 1 * max(L+bb_num)
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        # mask 为 0的部分, 赋值 -10000.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        # if img_encoded_layers.shape[1] != batch['img_len']:
        #     import pdb;pdb.set_trace()
        # if (img_emb - img_encoded_layers).abs().sum() + (txt_emb - txt_encoded_layers).abs().sum() > 0.01:
        #     import pdb;pdb.set_trace()
        #txt_img_embedding = self.gather_embeddings(txt_emb=txt_emb, img_emb=img_emb, gather_index=gather_index)
        txt_img_embedding = self.gather_embeddings(txt_emb=txt_encoded_layers, img_emb=img_encoded_layers, gather_index=gather_index)

        fusion_encoded_layers = self.encoder_fusion(
            txt_img_embedding, extended_attention_mask,
            output_all_encoded_layers=False
        )
        fusion_encoded_layers = fusion_encoded_layers[-1]
        return fusion_encoded_layers

    def forward_itm_fusion(self, batch):
        pass


class UniterTwoFlowRetrievalModel(UniterPreTrainedModel):
    """ Modification for Joint Vision-Language Encoding
        Yida Zhao 2021/1/6
        Tow separate transformer encoder for txt and image
        One multimodal encoder to fuse txt and image output
    """

    def __init__(self, config, img_dim):
        super().__init__(config)
        assert config.model_mode == 'two_flow'
        self.frozen_txt_encoder_layer = config.frozen_txt_encoder_layer
        # embeddings = tokenEmd + PosiEmd, 不加 typeEmb
        self.embeddings = UniterTextEmbeddings(config)
        # img_embeddings = RegionEmd + PosEmd, 不加 typeEmb
        self.img_embeddings = UniterImageEmbeddings(config, img_dim)
        # anwen hu 2020/12/15
        # txt encoder is named as self.encoder to be consistent with the init uniter model(come from txt bert model)
        self.encoder = UniterEncoder(config, config.txt_encoder_layers)
        self.encoder_img = UniterEncoder(config, config.img_encoder_layers)
        self.encoder_fusion = UniterEncoder(config, config.fusion_encoder_layers)
        self.pooler = BertTwoFlowPooler(config)
        # add by zyd 2020.12.31
        self.img_pooler = BertTwoFlowPooler(config)
        self.txt_pooler = BertTwoFlowPooler(config)

        self.apply(self.init_weights)

    def _compute_txt_embeddings(self, input_ids, position_ids,
                                txt_type_ids=None):
        output = self.embeddings(input_ids, position_ids, txt_type_ids)
        return output

    def _compute_img_embeddings(self, img_feat, img_pos_feat, img_masks=None,
                                img_type_ids=None):
        img_type_embeddings = None
        output = self.img_embeddings(img_feat, img_pos_feat,
                                     img_type_embeddings, img_masks)
        return output

    def gather_embeddings(self, txt_emb, img_emb, gather_index):
        gather_index = gather_index.unsqueeze(-1).expand(-1, -1, self.config.hidden_size)
        embedding_output = torch.gather(torch.cat([txt_emb, img_emb], dim=1),
                                        dim=1, index=gather_index)
        return embedding_output

    def forward(self, batch, img_masks=None,
                output_all_encoded_layers=True,
                txt_type_ids=None, img_type_ids=None, fusion=True):
        '''
        output: txt_encoded_layers, img_encoded_layers, fusion_encoded_layers
        batch 内存放的通用的
        '''
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attention_mask = batch['attn_masks']
        attention_mask_txt = batch['attn_masks_txt']
        attention_mask_img = batch['attn_masks_img']
        gather_index = batch['gather_index']
        # compute self-attention mask
        if len(attention_mask.shape) == 3:  # only designed for captioning task in unified encoder-decoder
            # batch * max(L+bb_num) * max(L+bb_num) > batch * 1 * max(L+bb_num) * max(L+bb_num)
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            # batch * max(L+bb_num) > batch * 1 * 1 * max(L+bb_num)
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        # mask 为 0的部分, 赋值 -10000.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # txt_mask
        if input_ids is not None:
            if len(attention_mask.shape) == 3:  # only designed for captioning task in unified encoder-decoder
                # batch * max(L) * max(L) > batch * 1 * max(L) * max(L)
                extended_attention_mask_txt = attention_mask_txt.unsqueeze(1)
            else:
                # batch * max(L) > batch * 1 * 1 * max(L)
                extended_attention_mask_txt = attention_mask_txt.unsqueeze(1).unsqueeze(2)
            extended_attention_mask_txt = extended_attention_mask_txt.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask_txt = (1.0 - extended_attention_mask_txt) * -10000.0
        # img_mask
        if img_feat is not None:
            extended_attention_mask_img = attention_mask_img.unsqueeze(1).unsqueeze(2)
            extended_attention_mask_img = extended_attention_mask_img.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask_img = (1.0 - extended_attention_mask_img) * -10000.0

        if input_ids is None:
            # image only
            img_embd = self._compute_img_embeddings(img_feat, img_pos_feat, img_masks, img_type_ids)
            img_encoded_layers = self.encoder_img(img_embd, extended_attention_mask_img,
                                                  output_all_encoded_layers=output_all_encoded_layers)
            if not output_all_encoded_layers:
                img_encoded_layers = img_encoded_layers[-1]
            txt_encoded_layers = None
            fusion_encoded_layers = None
            # anwen hu 2020/12/10: only image, don't need to use encoder_fusion

            # add by zyd 2020.12.31
            img_encoded_layers = self.img_pooler(img_encoded_layers, attention_mask_img, dropout=True, norm=True)

            return txt_encoded_layers, img_encoded_layers, fusion_encoded_layers
        elif img_feat is None:
            # text only
            txt_embd = self._compute_txt_embeddings(input_ids, position_ids, txt_type_ids)
            txt_encoded_layers = self.encoder(txt_embd, extended_attention_mask_txt,
                                              frozen_en_layer=self.frozen_txt_encoder_layer,
                                              output_all_encoded_layers=output_all_encoded_layers)
            if not output_all_encoded_layers:
                txt_encoded_layers = txt_encoded_layers[-1]
            # anwen hu 2020/12/10: only txt, don't need to use encoder_fusion
            img_encoded_layers = None
            fusion_encoded_layers = None

            # add by zyd 2020.12.31
            txt_encoded_layers = self.txt_pooler(txt_encoded_layers, attention_mask_txt, dropout=True, norm=True)
            
            return txt_encoded_layers, img_encoded_layers, fusion_encoded_layers

        else:
            img_embd = self._compute_img_embeddings(img_feat, img_pos_feat, img_masks, img_type_ids)
            img_encoded_layers = self.encoder_img(img_embd, extended_attention_mask_img,
                                                  output_all_encoded_layers=output_all_encoded_layers)
            # print('model/model.py img_encoded_layers[-1]:', img_encoded_layers[-1].shape)
            txt_embd = self._compute_txt_embeddings(input_ids, position_ids, txt_type_ids)
            txt_encoded_layers = self.encoder(txt_embd, extended_attention_mask_txt,
                                              frozen_en_layer=self.frozen_txt_encoder_layer,
                                              output_all_encoded_layers=output_all_encoded_layers)
            # print('model/model.py txt_encoded_layers[-1]:', txt_encoded_layers[-1].shape)
            if fusion:  # whether to fuse txt and img feature in fusion encoders
                txt_img_embedding = self.gather_embeddings(txt_emb=txt_encoded_layers[-1], img_emb=img_encoded_layers[-1],
                                                           gather_index=gather_index)
                fusion_encoded_layers = self.encoder_fusion(
                    txt_img_embedding, extended_attention_mask,
                    output_all_encoded_layers=output_all_encoded_layers)
                if not output_all_encoded_layers:
                    txt_encoded_layers = txt_encoded_layers[-1]
                    img_encoded_layers = img_encoded_layers[-1]
                    fusion_encoded_layers = fusion_encoded_layers[-1]
                    # add by zyd 2021.1.6
                    fusion_encoded_layers = self.pooler(fusion_encoded_layers, attention_mask)
            else:
                if not output_all_encoded_layers:
                    txt_encoded_layers = txt_encoded_layers[-1]
                    img_encoded_layers = img_encoded_layers[-1]
                    fusion_encoded_layers = None
            
            # add by zyd 2020.12.31
            txt_encoded_layers = self.txt_pooler(txt_encoded_layers, attention_mask_txt, dropout=True, norm=True)
            img_encoded_layers = self.img_pooler(img_encoded_layers, attention_mask_img, dropout=True, norm=True)

            return txt_encoded_layers, img_encoded_layers, fusion_encoded_layers

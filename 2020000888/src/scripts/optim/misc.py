"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Misc lr helper
"""
from apex.amp.frontend import state_dict
from torch.optim import Adam, Adamax

from .adamw import AdamW


def build_optimizer(model, opts, ckpt=None):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': opts.weight_decay},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # currently Adam only
    if opts.optim == 'adam':
        OptimCls = Adam
    elif opts.optim == 'adamax':
        OptimCls = Adamax
    elif opts.optim == 'adamw':
        OptimCls = AdamW
    else:
        raise ValueError('invalid optimizer')
    optimizer = OptimCls(optimizer_grouped_parameters,
                         lr=opts.learning_rate, betas=opts.betas)
    if ckpt is not None:
        optimizer.load_state_dict(ckpt)
    return optimizer


# anwen hu 2020/12/30: separate parameters into decoder and nodecder to apply different lr at following step
def build_capdec_optimizer(model, opts):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    decoder = ['decoder']
    """nodecoder_decay_param_names = [n for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay) and not any(nd in n for nd in decoder)]
    print('nodecoder_decay_param_names:', nodecoder_decay_param_names)
    nodecoder_no_decay_param_names = [n for n, p in param_optimizer
                    if any(nd in n for nd in no_decay) and not any(nd in n for nd in decoder)]
    print('nodecoder_no_decay_param_names:', nodecoder_no_decay_param_names)
    decoder_decay_param_names= [n for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay) and any(nd in n for nd in decoder)]
    print('decoder_decay_param_names:', decoder_decay_param_names)
    decoder_no_decay_param_names = [n for n, p in param_optimizer
                    if any(nd in n for nd in no_decay) and any(nd in n for nd in decoder)]
    print('decoder_no_decay_param_names:', decoder_no_decay_param_names)"""
    # TODO: THE LayerNorm Name in DECODER
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay) and not any(nd in n for nd in decoder)],
         'weight_decay': opts.weight_decay,
         'name': 'nodecoder_decay'},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay) and not any(nd in n for nd in decoder)],
         'weight_decay': 0.0,
         'name': 'nodecoder_no_decay'},
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay) and any(nd in n for nd in decoder)],
         'weight_decay': opts.weight_decay,
         'name': 'decoder_decay'},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay) and any(nd in n for nd in decoder)],
         'weight_decay': 0.0,
         'name': 'decoder_no_decay'}
    ]
    # currently Adam only
    if opts.optim == 'adam':
        OptimCls = Adam
    elif opts.optim == 'adamax':
        OptimCls = Adamax
    elif opts.optim == 'adamw':
        OptimCls = AdamW
    else:
        raise ValueError('invalid optimizer')
    optimizer = OptimCls(optimizer_grouped_parameters,
                         lr=opts.learning_rate, betas=opts.betas)
    return optimizer



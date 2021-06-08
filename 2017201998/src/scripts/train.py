import os
import sys
import torch
import json
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as Optim
from torch.autograd import Variable

import utils
import dataset
import modules_gru
import modules_trans

from eval import Evaluator
sys.path.append("./metric")
from metric import eval_utils

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

parser = argparse.ArgumentParser(description='train.py')
# hyperparameter
parser.add_argument('-n_embs', type=int, default=512, help='Embedding size')
parser.add_argument('-dim_ff', type=int, default=512, help='Feed forward hidden size')

parser.add_argument('-batch_size', type=int, default=8, help='Batch Size')
parser.add_argument('-epoch', type=int, default=100, help='Number of Epoch')
parser.add_argument('-report', type=int, default=50, help='Report interval')
parser.add_argument('-dropout', type=float, default=0.3, help='Dropout Rate')
parser.add_argument('-input_max_len', type=int, default=200, help='Max length of captions')
parser.add_argument('-g_max_len', type=int, default=20, help='Max length of captions')
parser.add_argument('-lr', type=float, default=5e-4, help='Learning Rate')
parser.add_argument('-beam_size', type=int, default=1, help='Beam Size')

parser.add_argument('-data_path', type=str, default='/data2/cwj/Arxiv/data/')
parser.add_argument('-restore', type=str, default='', help="Restoring model path")
parser.add_argument('-valid_type', type=str, default='cider', help='Validation method')
parser.add_argument('-out_path', type=str, default='./checkpoint', help='out path')
parser.add_argument('-out_file', type=str, default=None, help = 'result for evaluation')
parser.add_argument('-mode', type=str, default=None)
parser.add_argument('-model', type=str, default='G2G')
args = parser.parse_args()

# random seed
#torch.manual_seed(1234)
#torch.cuda.manual_seed(1234)

if args.valid_type is None:
    print('Please enter validation method ')
    sys.exit()

# load dict
vocabs, rev_vocabs = utils.load_vocabs(args.data_path)
vocab_size = len(vocabs)+1

if not os.path.exists(args.out_path):
    os.mkdir(args.out_path)

def save_model(path, model):
    model_state_dict = model.state_dict()
    torch.save(model_state_dict, path)

def get_dataset(data_path, set_type=None, vocabs=vocabs, is_train=True):
    return dataset.Dataset(data_path,
                           vocabs = vocabs,
                           rev_vocabs = rev_vocabs,
                           g_max_len = args.g_max_len,
                           input_max_len = args.input_max_len,
                           set_type = set_type)

def get_dataloader(dataset, batch_size, is_train=True):
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=is_train)

def train():
    train_set = get_dataset(os.path.join(args.data_path, 'train.json'), None)
    valid_set = get_dataset(os.path.join(args.data_path, 'val.json'), args.valid_type)

    train_batch = get_dataloader(train_set, args.batch_size, is_train=True)

    if args.model == 'G2G':
        model = modules_gru.Model(n_embs = args.n_embs,
                                  n_hidden = args.dim_ff,
                                  vocab_size = vocab_size,
                                  dropout = args.dropout,
                                  g_max_len = args.g_max_len)
    elif args.model == 'T2G':
        model = modules_trans.Model(n_embs = args.n_embs,
                                    n_hidden = args.dim_ff,
                                    vocab_size = vocab_size,
                                    dropout = args.dropout,
                                    g_max_len = args.g_max_len,
                                    input_max_len = args.input_max_len)

    #model = nn.DataParallel(model)
    if args.restore != '':
        model_dict = torch.load(args.restore)
        model.load_state_dict(model_dict)

    model.cuda()
    optim = Optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr) #, weight_decay=0.01)
    # scheduler = Optim.lr_scheduler.StepLR(optim, step_size=4, gamma=0.9)
    best_score = -100000000
    no_beat = 0
    for i in range(args.epoch):
        model.train()
        report_loss, start_time, n_samples = 0, time.time(), 0
        count = 0
        total = len(train_set) // args.batch_size + 1

        for batch in train_batch:
            model.zero_grad()
            abstract, title = batch
            abstract = Variable(abstract).cuda()
            title = Variable(title).cuda()

            loss = model(abstract, title)

            loss.backward()
            optim.step()

            report_loss += loss.data.item()
            n_samples += len(abstract.data)
            count += 1

            if count % args.report == 0 or count == total:
                # report loss
                print('%d/%d, epoch: %d, report_loss: %.3f, time: %.2f'
                      % (count, total, i+1, report_loss / n_samples, time.time() - start_time))
                if i >= 0:
                    if args.valid_type == 'P':
                        score = eval_p(valid_set, model)
                    else:
                        score = eval(valid_set, model)
                    model.train()

                    if score > best_score:
                        no_beat = 0
                        best_score = score
                        print('Score Beat ', score, '\n')
                        save_model(os.path.join(args.out_path, 'best_checkpoint_gru_25.pt'), model)
                    else:
                        no_beat += 1
                        save_model(os.path.join(args.out_path, 'checkpoint_gru_25.pt'), model)
                        print('Term ', no_beat, 'Best_score', best_score, '\n')
                        if no_beat == 50:
                            sys.exit()
                    report_loss, start_time, n_samples = 0, time.time(), 0
        #scheduler.step()
        print('Learning Rate ', optim.state_dict()['param_groups'][0]['lr'])
    return 0

def eval_p(dev_set, model):
    print('Starting Evaluation...')
    start_time = time.time()
    model.eval()
    dev_batch = get_dataloader(dev_set, args.batch_size, is_train=False)
    loss = 0
    with torch.no_grad():
        for batch in dev_batch:
            abstract, title = batch
            abstract = Variable(abstract).cuda()
            title = Variable(title).cuda()

            loss += torch.sum(model(abstract, title)).item()
    print('Loss: ', loss / len(dev_set))
    print("evaluting time:", time.time() - start_time)
    return -loss / len(dev_set)

def eval(dev_set, model):
    print('Starting Evaluation...')
    start_time = time.time()
    model.eval()
    loss = 0
    candidates = {}
    references = {}
    with torch.no_grad():
        for i in range(len(dev_set)):
            paperId, abstract, title = dev_set.get_data(i)
            abstract = Variable(abstract).cuda()
            ids = model.generate(abstract, beam_size=args.beam_size).data[0].tolist()
            sentences = transform(ids)
            
            if i < 6:
                print('   ', ' '.join(sentences))

            candidates[paperId] = [' '.join(sentences)]
            references[paperId] = [' '.join(title)]

    evaluator = Evaluator(references, candidates)
    score = evaluator.evaluate()
    print('BLUE-4: ', score)
    print("evaluting time:", time.time() - start_time)
    return score

def test():
    test_set = get_dataset(os.path.join(args.data_path, 'test.json'))

    if args.model == 'G2G':
        model = modules_gru.Model(n_embs = args.n_embs,
                                  n_hidden = args.dim_ff,
                                  vocab_size = vocab_size,
                                  dropout = args.dropout,
                                  g_max_len = args.g_max_len)
    elif args.model == 'T2G':
        model = modules_trans.Model(n_embs = args.n_embs,
                                    n_hidden = args.dim_ff,
                                    vocab_size = vocab_size,
                                    dropout = args.dropout,
                                    g_max_len = args.g_max_len,
                                    input_max_len = args.input_max_len)

    model_dict = torch.load(args.restore)
    model.load_state_dict({k.replace('module.', ''):v for k,v in model_dict.items()})
    model.cuda()
    model.eval()
    candidates, references, paperids = [], [], []

    with torch.no_grad():
        #with open(args.out_file, 'w', encoding='utf-8') as fout:
        for i in range(len(test_set)):
            paperId, abstract, title = test_set.get_data(i)
            abstract = Variable(abstract).cuda()
            ids = model.generate(abstract, beam_size=args.beam_size).data[0].tolist()
                
            candidates.append(transform(ids))
            references.append(' '.join(title))
            paperids.append(paperId)

    print('Total case ', len(candidates))
    data1=open("generate_result.txt",'w')
    for i in range(len(candidates)):
        print('img_id:',paperids[i],"\n",end='',file=data1)
        print('candidates:',candidates[i],"\n",end='',file=data1)
        print('references:',references[i],"\n\n",end='',file=data1)
    data1.close()
    result = eval_utils.eval_result(candidates, references, is_valid=False)    

def transform(ids):
    sentences = []
    for wid in ids:
        if wid == vocabs['<BOS>']:
            continue
        if wid == vocabs['<EOS>']:
            break
        sentences.append(rev_vocabs[wid])
    return sentences


if __name__ == '__main__':
    if args.mode == 'train':
        print('------Train Mode------')
        train()
    elif args.mode == 'test':
        print('------Test  Mode------')
        test()
    else:
        print('Please enter the mode')

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import log, euclidean_metric, count_acc, setup_seed
from data_loader import CategoriesSampler
#from transformer import *
from attention_module import *

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', default='0,1', type=str)
parser.add_argument('--load', default='/disks/sdc/yizhao_gao/ours/exp_mini/con_5shot/SSL1_con1_attn_T100_tem256_lr0.0005', type=str)
parser.add_argument('--cnn', default='ResNet12', type=str, help='ResNet12, Conv4Blocks')
parser.add_argument('--batch', default=2000, type=int)
parser.add_argument('--n_head', default=1, type=int)
parser.add_argument('--way', default=5, type=int)
parser.add_argument('--shot', default=5, type=int)
parser.add_argument('--query', default=15, type=int)
parser.add_argument('--dataset', default='miniImageNet', type=str)
parser.add_argument('--data_root', default='/disks/sdc/yizhao_gao/icml20/mini-imagenet/images', type=str)
parser.add_argument('--pretrained', default='/home/nanyi_fei/lab/iclr20/ours/pretrain/Res12-pre-mini.pth', type=str)
parser.add_argument('--test_seed', default=111, type=int)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
log_file_path = os.path.join(args.load, 'test_log.txt')
log(log_file_path, str(vars(args)))
setup_seed(args.test_seed)

# get dataloader
if args.dataset == 'miniImageNet':
    from data_loader import MiniImageNet as Dataset
elif args.dataset == 'CUB':
    from data_loader import CUB as Dataset
elif args.dataset == 'tieredImageNet':
    from data_loader import TieredImageNet as Dataset
test_set = Dataset(root=args.data_root, dataset=args.dataset, mode='test',cnn = args.cnn)
test_sampler = CategoriesSampler(test_set.label, args.batch, args.way, args.shot + args.query)
test_loader = DataLoader(dataset=test_set, batch_sampler=test_sampler, num_workers=8, pin_memory=True)

if args.cnn == 'ResNet12':
    from networks.res12 import ResNet
    base_net = ResNet().cuda()
    fea_dim = 640
    '''pretrained_dict = torch.load(args.pretrained)['params']
    model_dict = base_net.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k[8:]: v for k, v in pretrained_dict.items() if k[8:] in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    base_net.load_state_dict(model_dict)'''
elif args.cnn == 'Conv4Blocks':
    from networks.convnet import ConvNet
    base_net = ConvNet().cuda()
    fea_dim = 64

attn_net = MultiHeadAttention(args.n_head, fea_dim, fea_dim, fea_dim, dropout=0.5).cuda()
base_net = nn.DataParallel(base_net, device_ids=[0, 1])
attn_net = nn.DataParallel(attn_net, device_ids=[0, 1])
saved_models = torch.load(args.load + '/best_model.pth.tar')
base_net.load_state_dict(saved_models['base_net'])
attn_net.load_state_dict(saved_models['attn_net'])
base_net.eval()
attn_net.eval()

n_all = (args.shot+args.query) * args.way
label = torch.arange(args.way).repeat(args.query)
label = label.type(torch.cuda.LongTensor)
test_accuracies = []
with torch.no_grad():
    for i, batch in enumerate(test_loader, 1):
        inputs = batch[0].cuda() # [100, 3, 80, 80]
        inputs_aug1 = batch[1].cuda()
        inputs_aug2 = batch[2].cuda()
        inputs_aug3 = batch[3].cuda()
        features = base_net(inputs).unsqueeze(0) # [100, 640]
        features_aug1 = base_net(inputs_aug1).unsqueeze(0)
        features_aug2 = base_net(inputs_aug2).unsqueeze(0)
        features_aug3 = base_net(inputs_aug3).unsqueeze(0)
        #features_aug4 = base_net(inputs_aug4)
        fea_all = torch.cat((features,features_aug1,features_aug2,features_aug3),dim = 0)
        fea_all = fea_all.transpose(0,1)
        #fea_all = attn_net(fea_all)
        fea_all = attn_net(fea_all,fea_all,fea_all)
        fea = fea_all.reshape(n_all,-1)
        fea_shot, fea_query = fea[:args.way*args.shot], fea[args.way*args.shot:]
        # fea_shot: [25, 640]
        # fea_query: [75, 640]
        proto = fea_shot.reshape(args.shot, args.way, -1).mean(dim = 0)
        logits = euclidean_metric(fea_query, proto)
        acc = count_acc(logits, label)
        test_accuracies.append(acc)
        
        if i % 50 == 0:
            avg = np.mean(np.array(test_accuracies))
            std = np.std(np.array(test_accuracies))
            ci95 = 1.96 * std / np.sqrt(i + 1)
            log_str = 'batch {}: Accuracy: {:.4f} +- {:.4f} % ({:.4f} %)'.format(i, avg, ci95, acc)
            log(log_file_path, log_str)

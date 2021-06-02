import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
import torch.optim as optim
from torch.autograd import Variable
from utils import log, euclidean_metric, count_acc
from data_loader import CategoriesSampler
from networks.adversarial_net import AdvNet
#from transformer import *
from attention_module import *

def get_dataloaders(args):
    if args.dataset == 'miniImageNet':
        from data_loader import MiniImageNet as Dataset
    elif args.dataset == 'CUB':
        from data_loader import CUB as Dataset
    elif args.dataset == 'tieredImageNet':
        from data_loader import TieredImageNet as Dataset

    train_set = Dataset(root=args.data_root, dataset=args.dataset, mode='train',cnn = args.cnn)
    train_sampler = CategoriesSampler(train_set.label, args.iters_per_epoch, args.way, args.shot + args.query)
    train_loader = DataLoader(dataset=train_set, batch_sampler=train_sampler, num_workers=8, pin_memory=True)

    val_set = Dataset(root=args.data_root, dataset=args.dataset, mode='val',cnn = args.cnn)
    val_sampler = CategoriesSampler(val_set.label, args.val_episodes, args.val_way, args.val_shot + args.val_query)
    val_loader = DataLoader(dataset=val_set, batch_sampler=val_sampler, num_workers=8, pin_memory=True)

    return train_loader, val_loader

def get_networks(args):
    if args.cnn == 'ResNet12':
        from networks.res12 import ResNet
        base_net = ResNet().cuda()
        pretrained_dict = torch.load(args.pretrained)['params']
        model_dict = base_net.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k[8:]: v for k, v in pretrained_dict.items() if k[8:] in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        base_net.load_state_dict(model_dict)
        fea_dim = 640
    elif args.cnn == 'Conv4Blocks':
        from networks.convnet import ConvNet
        base_net = ConvNet().cuda()
        pretrained_dict = torch.load(args.pretrained)['params']
        model_dict = base_net.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        base_net.load_state_dict(model_dict)
        fea_dim = 64
    attn_net = MultiHeadAttention(args.n_head, fea_dim, fea_dim, fea_dim, dropout=0.5).cuda()
    proj_net = proj().cuda()
    return base_net,attn_net,proj_net,fea_dim

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', default='0,1', type=str)
parser.add_argument('--cnn', default='ResNet12', type=str)
parser.add_argument('--optimizer_type', default='sgd', type=str)
parser.add_argument('--max_epoch', default=100, type=int)
parser.add_argument('--iters_per_epoch', default=100, type=int)
parser.add_argument('--step_size', default=40, type=int)
parser.add_argument('--n_head', default=1, type=int)
parser.add_argument('--mix', default=0, type=int)
parser.add_argument('--proj', default=1, type=int)
parser.add_argument('--k', default=6, type=int)
parser.add_argument('--init_lr', default=0.0001, type=float)
parser.add_argument('--lambda_con', default=0.1, type=float)
parser.add_argument('--lambda_fsl', default=1, type=float)
parser.add_argument('--positive_mix', default=1, type=float)
parser.add_argument('--negtive_mix', default=1, type=float)
parser.add_argument('--dataset', default='miniImageNet', type=str)
parser.add_argument('--data_root', default='/disks/sdc/yizhao_gao/icml20/mini-imagenet/images', type=str)
parser.add_argument('--output_dir', default='./exp_mini_conv4/attn_5shot/mix_attn_con1_tem32_lr0.0005', type=str)
parser.add_argument('--shot', default=5, type=int)
parser.add_argument('--query', default=15, type=int)
parser.add_argument('--way', default=5, type=int)
parser.add_argument('--val_way', default=5, type=int)
parser.add_argument('--val_shot', default=5, type=int)
parser.add_argument('--val_query', default=15, type=int)
parser.add_argument('--val_episodes', default=600, type=int)
parser.add_argument('--temperature', default=256, type=int)
parser.add_argument('--T', default=1, type=float)
parser.add_argument('--pretrained', default='/disks/sdc/yizhao_gao/ours/pretrain/Res12-pre-mini.pth', type=str)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)
log_file_path = os.path.join(args.output_dir, 'train_log.txt')
log(log_file_path, str(vars(args)))

# get dataloaders
train_loader, val_loader = get_dataloaders(args)
# get networks
base_net,attn_net,proj_net,fea_dim = get_networks(args)
# set optimizer
param_groups = [{'params':base_net.parameters()},\
                {'params':attn_net.parameters(), 'lr':args.init_lr},\
                {'params':proj_net.parameters(), 'lr':args.init_lr}]
base_net = nn.DataParallel(base_net, device_ids=[0, 1])
proj_net = nn.DataParallel(proj_net, device_ids=[0, 1])
attn_net = nn.DataParallel(attn_net, device_ids=[0, 1])
# set optimizer
if args.optimizer_type == 'sgd':
    optimizer = torch.optim.SGD(param_groups, lr=args.init_lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
elif args.optimizer_type == 'adam':
    optimizer = torch.optim.Adam(param_groups, lr=args.init_lr)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)

# preparations
ce_loss = nn.CrossEntropyLoss()

n_shot_val = args.val_shot * args.val_way # 25
n_shot = args.shot * args.way # 150
n_all = (args.shot+args.query)*args.way

label_fsl_s = torch.arange(args.way).repeat(args.query) # 300
label_fsl_s = label_fsl_s.type(torch.cuda.LongTensor)
label_val = torch.arange(args.val_way).repeat(args.val_query) # 75
label_val = label_val.type(torch.cuda.LongTensor)


# train
print('start training...')
len_train_loader = len(train_loader)
print('len_train_loader:', len_train_loader) # 100
best_acc = 0.0
for epoch in range(1, args.max_epoch + 1):
    base_net.train()
    proj_net.train()
    attn_net.train()
    for i, batch in enumerate(train_loader, 1):
        with torch.autograd.set_detect_anomaly(True):
            inputs = batch[0].cuda() # [100, 3, 80, 80]
            inputs_aug1 = batch[1].cuda()
            inputs_aug2 = batch[2].cuda()
            inputs_aug3 = batch[3].cuda()
            features = base_net(inputs).unsqueeze(0) # [100, 640]
            features_aug1 = base_net(inputs_aug1).unsqueeze(0)
            features_aug2 = base_net(inputs_aug2).unsqueeze(0)
            features_aug3 = base_net(inputs_aug3).unsqueeze(0)
            #mixfea
            fea_all = torch.cat((features,features_aug1,features_aug2,features_aug3),dim = 0)
            fea_all2 = torch.cat((features,features_aug2,features_aug3,features_aug1),dim = 0)
            fea_all = fea_all.transpose(0,1)
            fea_all2 = fea_all2.transpose(0,1)
            fea_all = attn_net(fea_all,fea_all,fea_all)
            fea_all2 = attn_net(fea_all2,fea_all2,fea_all2)
            fea = fea_all.reshape(n_all,-1)
            fea2 = fea_all2.reshape(n_all,-1)
            if args.mix == 1:
                fea_shot, fea_query = fea[:n_shot], fea2[n_shot:]
            else:
                fea_shot, fea_query = fea[:n_shot], fea[n_shot:]
            # fea_shot: [25, 640]
            # fea_query: [75, 640]
            proto = fea_shot.reshape(args.shot, args.way, -1).mean(dim = 0) # [5, 640]
            logits = euclidean_metric(fea_query, proto)/args.temperature #[75, 5]
            fsl_loss = ce_loss(logits,label_fsl_s)
            acc = count_acc(logits, label_fsl_s)
            #con_loss
            con_loss = 0
            if args.lambda_con > 0:
                similarity_f = nn.CosineSimilarity()
                if args.proj == 1:
                    fea = proj_net(fea)
                fea_shot, fea_query = fea[:n_shot], fea[n_shot:]
                fea_query2 = fea2[n_shot:]
                proto = fea_shot.reshape(args.shot, args.way, -1).mean(dim = 0) # [5, 640]
                ind = torch.arange(args.query)
                for index in range(args.way):
                    p = proto[index].unsqueeze(0).repeat(args.way*args.query,1)#[75, 640]
                    s = similarity_f(p,fea_query)/args.T#[75]
                    s_sim = s.reshape(args.query,-1).t()#[5,15]
                    s2 = similarity_f(p,fea_query2)/args.T
                    s_sim2 = s2.reshape(args.query,-1).t()
                    l = 0
                    for index_j in range(args.query):
                        s_dif = 0
                        for index_m in range(args.way):
                            random.shuffle(ind)
                            ii = ind[0:args.k]
                            if index_m == index:
                                if args.positive_mix > 0:
                                    s_dif += torch.exp(s_sim2[index][index_j])
                                else:
                                    s_dif += torch.exp(s_sim[index][index_j])
                            else:
                                if args.negtive_mix > 0:
                                    s_dif += torch.exp(s_sim2[index_m][ii]).sum()
                                else:
                                    s_dif += torch.exp(s_sim[index_m][ii]).sum()
                        if args.positive_mix > 0:
                            l += -torch.log(torch.exp(s_sim2[index][index_j])/s_dif)
                        else:
                            l += -torch.log(torch.exp(s_sim[index][index_j])/s_dif)
                    con_loss += l/(args.way*args.k)
                con_loss /= args.way

            # total loss
            total_loss = args.lambda_fsl * fsl_loss + args.lambda_con * con_loss
            if args.lambda_con > 0:
                if i % 10 == 0:
                    print('iter:', i,'total_loss:', total_loss.item(),'fsl_loss:',fsl_loss.item(),'con_loss',con_loss.item())
                    print('iter:', i, 'fsl_acc:', acc)
            else:
                if i % 10 == 0:
                    print('iter:', i,'total_loss:', total_loss.item(),'fsl_loss:',fsl_loss.item())
                    print('iter:', i, 'fsl_acc:', acc)                

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
    lr_scheduler.step()

    # validation
    base_net.eval()
    attn_net.eval()
    ave_acc = []
    with torch.no_grad():
        for i_val, batch in enumerate(val_loader, 1):
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
            fea_all = attn_net(fea_all,fea_all,fea_all)
            fea = fea_all.reshape(n_all,-1)
            fea_shot, fea_query = fea[:n_shot_val], fea[n_shot_val:]
            # fea_shot: [25, 640]
            # fea_query: [75, 640]
            proto = fea_shot.reshape(args.val_shot, args.val_way, -1).mean(dim = 0)
            logits = euclidean_metric(fea_query, proto)
            acc = count_acc(logits, label_val)
            ave_acc.append(acc)
    ave_acc = np.mean(np.array(ave_acc))
    print('epoch {}: {:.2f}({:.2f})'.format(epoch, ave_acc * 100, acc * 100))
    if ave_acc > best_acc:
        best_acc = ave_acc
        torch.save({'base_net':base_net.state_dict(),'attn_net':attn_net.state_dict()}, os.path.join(args.output_dir, 'best_model.pth.tar'))
    log_str = "epoch: {:05d}, accuracy: {:.5f}".format(epoch, ave_acc)
    log(log_file_path, log_str)

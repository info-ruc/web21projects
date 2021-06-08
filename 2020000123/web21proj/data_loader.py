import torch
import numpy as np
import os
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class CategoriesSamplerOurs():
    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        print('max label:', max(label))
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch1 = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls*2]
            for c in classes[:self.n_cls]:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per*2]
                batch1.append(l[pos])
            batch1 = torch.stack(batch1).t().reshape(-1)

            batch2 = []
            for c in classes[self.n_cls:]:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch2.append(l[pos])
            batch2 = torch.stack(batch2).t().reshape(-1)
            
            batch = torch.cat([batch1, batch2])
            yield batch


class CategoriesSampler():
    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        print('max label:', max(label))
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch


class MiniImageNet(Dataset):
    def __init__(self, root, dataset='miniImageNet', mode='train', cnn='ResNet12'):
        csv_path = os.path.join('dataset', dataset, mode + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = os.path.join(root, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label
        self.num_class = len(set(label))

        if cnn == 'ResNet12':
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x/255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
                                     np.array([x/255.0 for x in [70.68188272, 68.27635443, 72.54505529]]))
            ])
            self.transform_aug1 = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(p=1),
                #transforms.RandomRotation((90,90)),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
                                     np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]))
            ])
            self.transform_aug2 = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.RandomVerticalFlip(p=1),
                #transforms.RandomRotation((90,90)),
                #transforms.RandomRotation((180,180)),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
                                     np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]))
            ])
            self.transform_aug3 = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.RandomRotation((270,270)),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
                                     np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]))
            ])
            
        else:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])
            self.transform_aug1 = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(p=1),
                #transforms.RandomRotation((90,90)),
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])
            self.transform_aug2 = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.RandomVerticalFlip(p=1),
                #transforms.RandomRotation((90,90)),
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])
            self.transform_aug3 = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.RandomRotation((270,270)),
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        raw_image = Image.open(path).convert('RGB')
        return self.transform(raw_image), self.transform_aug1(raw_image),self.transform_aug2(raw_image), self.transform_aug3(raw_image),label

    def __len__(self):
        return len(self.data)


class CUB(Dataset):
    def __init__(self, root, dataset='CUB', mode='train', cnn='ResNet12'):
        csv_path = os.path.join('dataset', dataset, mode + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = os.path.join(root, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label
        self.num_class = len(set(label))

        if cnn == 'ResNet12':
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x/255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
                                     np.array([x/255.0 for x in [70.68188272, 68.27635443, 72.54505529]]))
            ])
            self.transform_aug1 = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(p=1),
                #transforms.RandomRotation((90,90)),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
                                     np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]))
            ])
            self.transform_aug2 = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.RandomVerticalFlip(p=1),
                #transforms.RandomRotation((90,90)),
                #transforms.RandomRotation((180,180)),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
                                     np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]))
            ])
            self.transform_aug3 = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.RandomRotation((270,270)),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
                                     np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]))
            ])
            
        else:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])
            self.transform_aug1 = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(p=1),
                #transforms.RandomRotation((90,90)),
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])
            self.transform_aug2 = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.RandomVerticalFlip(p=1),
                #transforms.RandomRotation((90,90)),
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])
            self.transform_aug3 = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.RandomRotation((270,270)),
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        raw_image = Image.open(path).convert('RGB')
        return self.transform(raw_image), self.transform_aug1(raw_image),self.transform_aug2(raw_image), self.transform_aug3(raw_image),label

    def __len__(self):
        return len(self.data)


class TieredImageNet(Dataset):
    def __init__(self, root, dataset='tieredImageNet', mode='train', cnn='ResNet12'):
        csv_path = os.path.join('dataset', dataset, mode + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = os.path.join(root, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label
        self.num_class = len(set(label))

        if cnn == 'ResNet12':
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x/255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
                                     np.array([x/255.0 for x in [70.68188272, 68.27635443, 72.54505529]]))
            ])
            self.transform_aug1 = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(p=1),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
                                     np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]))
            ])
            self.transform_aug2 = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.RandomVerticalFlip(p=1),
                #transforms.RandomRotation((90,90)),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
                                     np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]))
            ])
            self.transform_aug3 = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.RandomRotation((270,270)),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
                                     np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]))
            ])
            
        else:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])
            self.transform_aug1 = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(p=1),
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])
            self.transform_aug2 = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.RandomVerticalFlip(p=1),
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])
            self.transform_aug3 = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.RandomRotation((270,270)),
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        raw_image = Image.open(path).convert('RGB')
        return self.transform(raw_image), self.transform_aug1(raw_image),self.transform_aug2(raw_image), self.transform_aug3(raw_image),label

    def __len__(self):
        return len(self.data)
import os
import torch

import clip
from PIL import Image
from tqdm import tqdm

from torchvision.utils import save_image

import random

root = '../outputs/finetuneOpenai'

target = '../results_top1/finetuneOpenai'

if not os.path.exists(target) : 
    os.mkdir(target)
    os.mkdir(os.path.join(target, 'img'))
    os.mkdir(os.path.join(target, 'txt'))

folds = os.listdir(root)

print("total files : ", len(folds))

results = '../results_top1/finetuneVAE_finetuneTrans/img/'

results_name = os.listdir(results)

temp = []

for fold in folds : 
    if fold in results_name : 
        temp.append(fold)

folds = temp

names = clip.available_models()

print(names)

device = 'cuda'

model, preprocess = clip.load("ViT-B/32", device=device)

for fold in tqdm(folds) : 
    files = os.listdir(os.path.join(root, fold))
    
    imgs = []
    txt = []
    origin = []
    
    for file in files : 
        if file.split('.')[-1] == 'txt' : 
            txt.append(file)
        elif file.split('.')[-1] == 'jpg' : 
            if file == 'origin.jpg' : 
                origin = file
                continue
            imgs.append(file)
        else : 
            print("file type Error")
            exit()
    
    images = []
    for img in imgs : 
        img_name = os.path.join(root, fold, img)
        image = preprocess(Image.open(img_name)).to(device)
        images.append(image)
    
    images = torch.stack(images, 0)
    
    origin_name = os.path.join(root, fold, origin)
    origin_img = image = preprocess(Image.open(origin_name)).unsqueeze(0).to(device)
    
    if len(txt) > 1 : 
        print("here are multiple txt files")
        exit()
    else : 
        with open(os.path.join(root, fold, txt[0]), 'r+') as f : 
            sens = f.readlines()
            sen = sens[-1]
            sen = sen.split('.')[0]
            #print(sen)
            text = clip.tokenize([sen]).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(images)
        text_features = model.encode_text(text)
        origin_features = model.encode_image(origin_img)
    
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    origin_features /= origin_features.norm(dim=-1, keepdim=True)
    similarity = text_features @ image_features.T
    gt = text_features @ origin_features.T
    indice = similarity.argmax(dim = -1)
    print(gt[0,0], similarity[0,indice])
    
    save_image(images[indice], os.path.join(target, 'img', fold), normalize=True)
    with open(os.path.join(target, 'txt', fold.split('.')[0] + '.txt') , 'a+') as f : 
        f.write(sen)

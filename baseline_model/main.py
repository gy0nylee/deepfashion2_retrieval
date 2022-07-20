import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
import pickle
import argparse

from dataset import TripletData, RetrievalData
from metric import TopkAccuracy

with open('true_idcs_list_val.pickle','rb') as f:
    true_idcs_list_val = pickle.load(f)
with open('pairs_train.pickle','rb') as f:
    pairs_train = pickle.load(f)
with open('pairs_validation.pickle','rb') as f:
    pairs_validation = pickle.load(f)
with open('shop_idx_val.pickle', 'rb') as f:
    shop_idx_val = pickle.load(f)
with open('query_idx_val.pickle','rb') as f:
    query_idx_val = pickle.load(f)
with open('whole_images_train.pickle','rb') as f:
    whole_images_train = pickle.load(f)


# img_paths
train_paths = os.listdir(os.path.join('train', 'train', 'cropped'))
test_paths = os.listdir(os.path.join('validation', 'validation', 'cropped'))

parser = argparse.ArgumentParser(description='Baseline_model')
parser.add_argument('--batch_size', '-b', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--num_workers', '-nw',type=int, default=8)
parser.add_argument('--weight_decay', '-wd',type=float, default=0)
parser.add_argument('--optim', default=optim.Adam)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--k', type=int, default=20)
args = parser.parse_args()





# Dataloader
batch_size = 32
lr = 0.001
num_workers = 8
weight_decay = 0


transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5588843, 0.5189913,0.5125264),
                         std = (0.24053435,0.24247852,0.22778076))
])



train_ds_t = TripletData(pairs=pairs_train, transform=transforms, img_paths=train_paths, set='train')
val_ds_t = TripletData(pairs=pairs_validation, transform=transforms, img_paths=train_paths, set='train')
val_ds_q = RetrievalData(idx_set=query_idx_val, pairs=pairs_validation, transform=transforms, img_paths=train_paths, set='train')
val_ds_g = RetrievalData(idx_set=shop_idx_val, pairs=None, transform=transforms, img_paths=train_paths, set='train')



### model
model = models.resnet18().cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
triplet_loss = nn.TripletMarginLoss()


train_loader = DataLoader(train_ds_t, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)
val_loader = DataLoader(val_ds_t, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
val_loader_q = DataLoader(val_ds_q, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
val_loader_g = DataLoader(val_ds_g, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)


#### training
def train(epoch):
    model.train()
    train_loss = 0
    train_losses = []

    pbar = tqdm(train_loader)
    for batch, data in enumerate(pbar):
        optimizer.zero_grad()
        u, sp, sn = data
        output1 = model(u.cuda())
        output2 = model(sp.cuda())
        output3 = model(sn.cuda())
        loss = triplet_loss(output1, output2, output3)
        train_loss += loss.item()
        pbar.set_description(f'training - loss: {train_loss / (batch + 1)}')
        loss.backward()
        optimizer.step()
    train_losses.append(train_loss)
    return train_loss

def validate(epoch, k):
    val_loss = 0
    val_losses = []
    model.eval()
    with torch.no_grad():
        pbar = tqdm(val_loader)
        for batch, data in enumerate(pbar):
            u, sp, sn = data
            output1 = model(u.cuda())
            output2 = model(sp.cuda())
            output3 = model(sn.cuda())
            loss = triplet_loss(output1, output2, output3)
            val_loss += loss.item()
            pbar.set_description(f'validation - loss: {val_loss / (batch + 1)}')

        features = []
        for img in tqdm(val_loader_g, desc='extracting gallery features'):
            output = model(img.cuda())
            features.append(output.data)
        gallery_features = torch.cat(features)

        top_k_indices = []
        for img in tqdm(val_loader_q, desc='extracting query feature'):
            output = model(img.cuda())
            query_features = output.data

            # batch마다 (32:가능 64:불가능(OOM)) TOP-K 구하기 먼저 => TOP-K INDEX 출력
            cos = nn.CosineSimilarity(dim=-1)
            cos_sim = cos(query_features.unsqueeze(1), gallery_features)
            _, indices = torch.topk(cos_sim, k = k)
            top_k_indices.append(indices)
        top_k_indices = torch.cat(top_k_indices)
        topk_acc = TopkAccuracy(true_idcs_list_val, top_k_indices.cpu(), shop_idx_val)
    val_losses.append(val_loss)
    return val_loss, topk_acc

model_path = 'best_model.pth'
patience = 3

min_val_loss = np.inf
for epoch in range(args.epochs):
    p = patience
    train_loss = train(epoch)
    val_loss, topk_acc = validate(epoch, args.k)
    print(f'Epoch {epoch + 1} \t\t '
          f'Training Loss: {train_loss / len(train_loader)} \t\t '
          f'Validation Loss: {val_loss / len(val_loader)} \t\t'
          f'Validation TopkAcc: {topk_acc} \t\t')
    torch.save(model, model_path)




   ''' if min_val_loss > val_loss:
        min_val_loss = val_loss
        torch.save(model, model_path)
        p = patience
    else:
        p -= 1
        torch.save(model, model_path)
        if p == 0:
            print(f'Early Stopping. Min_val_loss : {min_val_loss}')
            break'''



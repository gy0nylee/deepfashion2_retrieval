import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
import pickle
import argparse
import wandb
import torchvision.models as models
import torch.cuda.amp as amp


from baseline_model.dataset import TripletData, RetrievalData
from baseline_model.metric import TopkAccuracy
from baseline_model.optimizer import get_optimizer
from baseline_model.models import get_model



#wandb.init()

with open('true_idcs_list_val.pickle','rb') as f:
    true_idcs_list_val = pickle.load(f)
with open('infos_train.pickle','rb') as f:
    infos_train = pickle.load(f)
with open('infos_val.pickle','rb') as f:
    infos_val = pickle.load(f)

img_mean = [0.5588843, 0.5189913,0.5125264]
img_std = [0.24053435,0.24247852,0.22778076]

# img_paths
train_paths = os.listdir(os.path.join('train', 'train', 'cropped'))
#test_paths = os.listdir(os.path.join('validation', 'validation', 'cropped'))


'''
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
'''






# args
parser = argparse.ArgumentParser(description='Baseline_model')
parser.add_argument('--batch_size', '-b', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--num_workers', '-nw',type=int, default=8)
parser.add_argument('--weight_decay', '-wd',type=float, default=5e-4)
parser.add_argument('--momentum', '-m',  type=float, default=0.9)
parser.add_argument('--eps', '-e', type=float, default=1e-8)
parser.add_argument('--optim',  type=str, default='adam')
parser.add_argument('--model',  type=str, default='resnet18')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--k', type=int, default=20)
parser.add_argument('--sampling', type=str, default='random')
args = parser.parse_args()




# Dataloader

train_ds_t = TripletData(infos_train, args)
val_ds_t = TripletData(infos_val, args)
val_ds_q = RetrievalData(source='user',infos = infos_val, img_paths=train_paths, folder='train')
val_ds_g = RetrievalData(source='shop',infos = infos_val, img_paths=train_paths, folder='train')




### model
torch.backends.cudnn.benchmark = True
model = get_model(args).cuda()
model = nn.DataParallel(model, device_ids=[0,1]).cuda()

for param in model.parameters():
    param.requires_grad = False
model.fc.weight.requires_grad = True
model.fc.bias.requires_grad = True


scaler = amp.GradScaler() # https://tutorials.pytorch.kr/recipes/recipes/amp_recipe.html
optimizer = get_optimizer(model.parameters(), args)
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
        with amp.autocast():
            output1 = model(u.cuda())
            output2 = model(sp.cuda())
            output3 = model(sn.cuda())
            loss = triplet_loss(output1, output2, output3)
        train_loss += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        pbar.set_description(f'training - loss: {train_loss / (batch + 1)}')
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

        features = []
        for img in tqdm(val_loader_q, desc='extracting query feature'):
            output = model(img.cuda())
            features = output.data
        query_features = torch.cat(features)

        # 32개마다 TOP-K 구하기 먼저 => TOP-K INDEX 출력
        top_k_indices = []
        query_chunks = torch.chunk(query_features, len(query_features) // 32 + 1)
        for i in range(len(query_features) // 32 + 1):
            cos = nn.CosineSimilarity(dim=-1)
            cos_sim = cos(query_chunks[i].unsqueeze(1), gallery_features)
            _, indices = torch.topk(cos_sim, k=k)
            top_k_indices.append(indices)
        top_k_indices = torch.cat(top_k_indices)
        topk_acc = TopkAccuracy(true_idcs_list_val, top_k_indices.cpu(), shop_idx_val)
    val_losses.append(val_loss)
    return val_loss, topk_acc


#model_path = 'best_model.pth'
#patience = 3

min_val_loss = np.inf
for epoch in range(args.epochs):
    #p = patience
    train_loss = train(epoch)
    val_loss, topk_acc = validate(epoch, args.k)
    wandb.log({"train_loss": train_loss/len(train_loader), "val_loss": val_loss/len(val_loader), "topk_acc": topk_acc}, step=epoch)
    print(f'Epoch {epoch + 1} \t\t '
          f'Training Loss: {train_loss / len(train_loader)} \t\t '
          f'Validation Loss: {val_loss / len(val_loader)} \t\t'
          f'Validation TopkAcc: {topk_acc} \t\t')
    #torch.save(model, model_path)




""" 
  if min_val_loss > val_loss:
        min_val_loss = val_loss
        torch.save(model, model_path)
        p = patience
    else:
        p -= 1
        torch.save(model, model_path)
        if p == 0:
            print(f'Early Stopping. Min_val_loss : {min_val_loss}')
            break
"""




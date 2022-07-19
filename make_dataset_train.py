import torchvision
import os
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader, random_split, default_convert
from torch.autograd import Variable
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import random
import pickle
#from collate_modified import modified_collate

import torch, gc
gc.collect()
torch.cuda.empty_cache()

with open('pairs_train.pickle','rb') as f:
    pairs_train = pickle.load(f)
with open('pairs_validation.pickle','rb') as f:
    pairs_validation = pickle.load(f)
#with open('pairs_test.pickle','rb') as f:
#    pairs_test = pickle.load(f)

with open('shop_idx_val.pickle', 'rb') as f:
    shop_idx_val = pickle.load(f)
#with open('shop_idx_test.pickle','rb') as f:
#    shop_idx_test = pickle.load(f)

with open('query_idx_val.pickle','rb') as f:
    query_idx_val = pickle.load(f)
#with open('query_idx_test.pickle','rb') as f:
#    query_idx_test = pickle.load(f)

with open('whole_images_train.pickle','rb') as f:
    whole_images_train = pickle.load(f)

'''
train_idx = [t['idx'] for t in tqdm(whole_images_train)]
class TempData(Dataset):
    def __init__(self, idx, transform, img_paths):
        self.idx = idx
        self.transform = transform
        self.img_paths = img_paths

    def __len__(self):
        return len(self.idx)
    def __getitem__(self, i):
        img = Image.open(os.path.join('train', 'train', 'cropped',self.img_paths[self.idx[i]]))
        if self.transform is not None:
            img = self.transform(img)
        return img

transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

train_paths = os.listdir(os.path.join('train', 'train', 'cropped'))

train_whole = TempData(idx=train_idx, transform=transforms, img_paths=train_paths)

mean_whole = [np.mean(x.numpy(), axis=(1,2)) for x in tqdm(train_whole)]
std_whole = [np.std(x.numpy(), axis=(1,2)) for x in tqdm(train_whole)]

mean_0 = np.mean([m[0] for m in mean_whole])
mean_1 = np.mean([m[1] for m in mean_whole])
mean_2 = np.mean([m[2] for m in mean_whole])

std_0 = np.mean([m[0] for m in std_whole])
std_1 = np.mean([m[1] for m in std_whole])
std_2 = np.mean([m[2] for m in std_whole])

print('mean:', mean_0,mean_1,mean_2)
print('std:', std_0,std_1,std_2)'''

#mean: 0.5588843 0.5189913 0.5125264
#std: 0.24053435 0.24247852 0.22778076



'''
print(len(whole_images_train))
print(len(pairs_train))
print(len(whole_images_validation))
print(len(pairs_validation))
'''

#parser = argparse.ArgumentParser()
#parser.add_argument( ~~~)
#args = parser.parse_args()



class TripletData(Dataset):
    def __init__(self, pairs, transform, img_paths, set):
        self.pairs = pairs
        self.transform = transform
        self.img_paths = img_paths
        self.set = set

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        user_img = Image.open(os.path.join(self.set, self.set, 'cropped',self.img_paths[self.pairs[i]['p'][0]]))
        shop_p_img = Image.open(os.path.join(self.set, self.set, 'cropped',self.img_paths[self.pairs[i]['p'][1]]))
        shop_n_img = Image.open(os.path.join(self.set, self.set, 'cropped',self.img_paths[self.pairs[i]['n'][1]]))


        if self.transform is not None:
            user_img = self.transform(user_img)
            shop_p_img = self.transform(shop_p_img)
            shop_n_img = self.transform(shop_n_img)

        return user_img, shop_p_img, shop_n_img

transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5588843, 0.5189913,0.5125264), std = (0.24053435,0.24247852,0.22778076))
])



class RetrievalData(Dataset):
    def __init__(self, idx_set ,source, pairs, transform, img_paths, set):
        self.idx_set = idx_set
        self.source = source
        self.pairs = pairs
        self.transform = transform
        self.img_paths = img_paths
        self.set = set

    def __len__(self):
        return len(self.idx_set)

    def __getitem__(self, i):
        img = Image.open(os.path.join(self.set, self.set, 'cropped', self.img_paths[self.idx_set[i]]))
        if self.transform is not None:
            img = self.transform(img)

        if self.source == 'user' :
            #true_idcs = [pair['p'][1] for pair in self.pairs if pair['p'][0] == self.idx_set[i]] # true indices
            #max_len = 82
            #padded_true_idcs = torch.tensor(true_idcs + [-1]*(max_len-len(true_idcs)))

            return img

        else:
            #gallery_idx = self.idx_set[i]
            return img


# img_paths
train_paths = os.listdir(os.path.join('train', 'train', 'cropped'))
test_paths = os.listdir(os.path.join('validation', 'validation', 'cropped'))


# Dataloader
batch_size = 32
lr = 0.001
num_workers = 8
weight_decay = 0


train_ds_t = TripletData(pairs=pairs_train, transform=transforms, img_paths=train_paths, set='train')
val_ds_t = TripletData(pairs=pairs_validation, transform=transforms, img_paths=train_paths, set='train')
val_ds_q = RetrievalData(idx_set=query_idx_val, source = 'user', pairs=pairs_validation, transform=transforms, img_paths=train_paths, set='train')
val_ds_g = RetrievalData(idx_set=shop_idx_val, source = 'shop', pairs=None, transform=transforms, img_paths=train_paths, set='train')

#length_list = [len(q) for _, q in tqdm(val_ds_q)]
#print(max(length_list))


#true_idcs_list = [[pair['p'][1] for pair in pairs_validation if pair['p'][0] == query_idx_val[i]] for i in tqdm(range(len(query_idx_val)))]
#with open('true_idcs_list_val.pickle','wb') as f:
#    pickle.dump(true_idcs_list, f)

with open('true_idcs_list_val.pickle','rb') as f:
    true_idcs_list_val = pickle.load(f)

'''
print(train_dataset[0])
temp_image = train_dataset[0][0].numpy()
temp_image = np.transpose(temp_image, (1,2,0))
plt.imshow(temp_image)
plt.savefig('temp_image.png')

temp = Image.open(os.path.join('train', 'train', 'cropped',train_paths[train_pairs[0]['p'][0]]))
temp.save('temp_image_og.png', 'png')
print('original_size',temp.size)
'''



### model
model = models.resnet18().cuda()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
triplet_loss = nn.TripletMarginLoss()


model_path = 'best_model.pth'
patience = 3


train_loader = DataLoader(train_ds_t, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)
val_loader = DataLoader(val_ds_t, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
#test_loader = DataLoader(test_ds_t, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

val_loader_q = DataLoader(val_ds_q, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
val_loader_g = DataLoader(val_ds_g, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

#test_loader_q = DataLoader(test_ds_q, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
#test_loader_g = DataLoader(test_ds_g, batch_size=batch_size, num_workers=num_workers, pin_memory=True)


def TopkAccuracy(true_idcs_list, top_k_indices, shop_idx_val):
    
    #query : 15437, gallery : 45455
    #true_idcs_list : query에 상응하는 true shop image의 index들 (train_path의 index, max_length로 (-1) padded) (15437,82) (Tensor)
    #top_k_indices : top_k (cos similarity) indices(0~45454) -> gallery에 대한 index (15437, k) (Tensor)
    #shop_idx_val : gallery의 true index list (train_path의 index) (45455)
    
    acc = 0
    for i in tqdm(range(len(true_idcs_list)), desc='calculating topkacc'):
        # true idcs중 하나라도 topk_idx_list에 있으면 +1
        if len(set(true_idcs_list[i]).intersection(np.array(shop_idx_val)[top_k_indices[i]])) > 0 :
           acc += 1
    return acc/len(true_idcs_list)



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

def validate(epoch, k=12):
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



epochs = 30
min_val_loss = np.inf
for epoch in range(epochs):
    p = patience
    train_loss = train(epoch)
    val_loss, topk_acc = validate(epoch, k=20)
    print(f'Epoch {epoch + 1} \t\t '
          f'Training Loss: {train_loss / len(train_loader)} \t\t '
          f'Validation Loss: {val_loss / len(val_loader)} \t\t'
          f'Validation TopkAcc: {topk_acc} \t\t')
    if min_val_loss > val_loss:
        min_val_loss = val_loss
        torch.save(model, model_path)
        p = patience
    else:
        p -= 1
        if p == 0:
            print(f'Early Stopping. Min_val_loss : {min_val_loss}')
            break



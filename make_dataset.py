import torchvision
import os
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader, random_split
from torch.autograd import Variable
from PIL import Image
from tqdm import tqdm
import os
import random
import pickle



with open('whole_images_train.pickle','rb') as f:
    whole_images_train = pickle.load(f)
with open('pairs_train.pickle','rb') as f:
    pairs_train = pickle.load(f)
with open('whole_images_validation.pickle','rb') as f:
    whole_images_validation = pickle.load(f)
with open('pairs_validation.pickle','rb') as f:
    pairs_validation = pickle.load(f)


print(len(whole_images_train))
print(len(pairs_train))
print(len(whole_images_validation))
print(len(pairs_validation))



class TripletData(Dataset):
    def __init__(self, pair, transform, img_paths):
        self.pair = pair
        self.transform = transform
        self.img_paths = img_paths

    def __len__(self):
        return len(self.pair)

    def __getitem__(self, i):
        user_img = Image.open(self.img_paths[pair[i]['p'][0]])
        shop_p_img = Image.open(self.img_paths[pair[i]['p'][1]])
        shop_n_img = Image.open(self.img_paths[pair[i]['n'][1]])

        if self.transform is not None:
            user_img = self.transform(user_img)
            shop_p_img = self.transform(shop_p_img)
            shop_n_img = self.transform(shop_n_img)

        return user_img, shop_p_img, shop_n_img

transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


# train-validation split
train_paths = os.listdir(os.path.join('train', 'train', 'cropped'))
test_paths = os.listdir(os.path.join('validation', 'validation', 'cropped'))

train_pairs, val_pairs = random_split(pairs_train.sample(50000, ignore_index=True), [40000, 10000])
test_pairs = pairs_validation

# Dataloader
batchi_size = 32
lr = 0.001
num_workers = 8
weight_decay = 0



train_dataset = TripletData(pair=train_pairs, transform=transforms, img_paths=train_paths)
val_dataset = TripletData(pair=val_pairs, transform=transforms, img_paths=train_paths)
test_dataset = TripletData(pair=test_pairs, transform=transforms, img_paths=test_paths)

train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)


device = 'cuda'
model = models.resnet18().cuda()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
triplet_loss = nn.TripletMarginLoss()

'''
def train(epoch):
    model.train()
    train_losses = []
    val_losses = []
    for data in tqdm(train_loader):
        optimizer.zero_grad()
        u, sp, sn = data

        e1 = model(u.to(device))
        e2 = model(sp.to(device))
        e3 = model(sn.to(device))

        loss = triplet_loss(e1, e2, e3)
        train_loss += loss
        loss.backward()
        optimizer.step()
    
    model.eval()
    for data in tqdm(val_loader):
        
    


for epoch in range(epochs):
print("Train Loss: {}".format(train_loss.item()), "Validation Loss: {}".format())
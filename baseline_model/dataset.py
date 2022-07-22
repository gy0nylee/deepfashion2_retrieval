from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from baseline_model.utils import get_pad
import random

train_paths = os.listdir(os.path.join('train', 'train', 'cropped'))
img_mean = [0.5588843, 0.5189913,0.5125264]
img_std = [0.24053435,0.24247852,0.22778076]


class TripletData(Dataset):
    def __init__(self, infos, args): # dict list 추가
        self.infos = infos
        self.img_paths = train_paths
        self.args = args
        self.mean = img_mean
        self.std = img_std

    def __len__(self):
        return len(self.infos[0])

    def __getitem__(self, i):
        anchor_img = Image.open(os.path.join('train', 'train', 'cropped',self.img_paths[self.infos[0][i][0]]))
        pos_img = Image.open(os.path.join('train', 'train', 'cropped',self.img_paths[self.infos[0][i][1]]))

        # negative sample 뽑기
        # (source, label, category dict) -> idx 추려내서 sampling
        # 1. source = shop , label != anchor label
        # 2. source = shop , label != anchor label, category = anchor category
        #  -> 1  or 2 - args.sampling
        neg_idx = self.sampling(i, self.img_paths, self.infos, self.args)
        neg_img = Image.open(os.path.join('train', 'train', 'cropped',self.img_paths[neg_idx[0]]))
        return self.aug(anchor_img), self.aug(pos_img), self.aug(neg_img)


    def aug(self, img):
        aug_transforms = transforms.Compose([
            transforms.Pad(get_pad(img), fill=0),
            transforms.RandomResizedCrop(size=(224,224), scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)])
        return aug_transforms(img)

    def sampling(self, i, img_paths, infos, args):

        anc = img_paths[infos[0][i][0]].split('_')
        anc_label = anc[0]+anc[1].zfill(2)  # anchor의 label
        anc_cate = anc[5][:-4] # anchor의 category
        in_label_idcs = set(infos[1]['shop']) - set(infos[2][anc_label]) # shop indices에서 label 같은 indices 소거

        if args.sampling == 'random': # anchor의 label을 제외한 나머지에서 1개 random sampling
            return random.sample(in_label_idcs,1)
        if args.sampling == 'category':
            in_cate_idcs = in_label_idcs & set(infos[3][anc_cate]) # category 같은 indices
            return random.sample(in_cate_idcs, 1)


class RetrievalData(Dataset):
    def __init__(self, source, infos, img_paths, folder):
        self.source = source # query='user', gallery='shop'
        self.infos = infos
        self.img_paths = img_paths # train_paths, test_paths
        self.folder = folder
        self.mean = img_mean
        self.std = img_std

    def __len__(self):
        return len(self.infos[1][self.source])

    def __getitem__(self, i):
        img = Image.open(os.path.join(self.folder, self.folder, 'cropped', self.img_paths[self.infos[1][self.source][i]]))
        return self.transform(img)

    def transform(self, img):
        trans = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)])
        return trans(img)




from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from baseline_model.utils import get_pad

img_mean = [0.5588843, 0.5189913,0.5125264]
img_std = [0.24053435,0.24247852,0.22778076]

class TripletData(Dataset):
    def __init__(self, pairs, transform, img_paths, set):
        self.pairs = pairs
        self.transform = transform
        self.img_paths = img_paths
        self.set = set
        self.mean = img_mean
        self.std = img_std

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        user_img = Image.open(os.path.join(self.set, self.set, 'cropped',self.img_paths[self.pairs[i]['p'][0]]))
        shop_p_img = Image.open(os.path.join(self.set, self.set, 'cropped',self.img_paths[self.pairs[i]['p'][1]]))

        # nagative sample 뽑기
        




        shop_n_img = Image.open(os.path.join(self.set, self.set, 'cropped',self.img_paths[self.pairs[i]['n'][1]]))
        return self.transform_tr(user_img), self.transform_tr(shop_p_img), self.transform_tr(shop_n_img)




    def transform_tr(self, img):
        aug_transforms = transforms.Compose([
            transforms.Pad(get_pad(img), fill=0),
            transforms.RandomResizedCrop(size=(224,224), scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)])
        return aug_transforms(img)




class RetrievalData(Dataset):
    def __init__(self, idx_set , pairs, transform, img_paths, set):
        self.idx_set = idx_set
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

        return img


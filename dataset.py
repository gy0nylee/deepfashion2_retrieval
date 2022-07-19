from torch.utils.data import Dataset, DataLoader
import os
import pickle

# image path
img_paths = os.listdir(os.path.join('train', 'train', 'cropped'))

# train/validation index
with open('train_idx.pickle','rb') as f:
    train_idx = pickle.load(f)
with open('validation_idx.pickle','rb') as f:
    validation_idx = pickle.load(f)


class DFdataset(Dataset):
    def __init__(self, idx, transform, img_paths, set):
        self.idx = idx
        self.img_paths = img_paths
        self.transform = transform
        self.set = set

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        img = Image.open(os.path.join(self.set, self.set, 'cropped',self.img_paths[self.idx[i]]))
        temp = self.img_paths[self.idx[i]].split('_')
        label = hash(temp[0] +'_'+ temp[1]) # 새로운 label 생성 -> hash를 label로 사용해도 되나...
        category = temp[5][:-4]
        source = temp[2]

        if self.transform is not None:
            img = self.transform(img)

        return img, label, category, source




transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5588843, 0.5189913,0.5125264), std = (0.24053435,0.24247852,0.22778076))
])

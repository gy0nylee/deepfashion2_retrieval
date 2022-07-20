from torch.utils.data import Dataset, DataLoader
import os
import pickle
from torchvision import transforms
from PIL import Image
from collections import Counter

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
        label = int(temp[0] + temp[1].zfill(2))
        category = temp[5][:-4]
        source = 1 if temp[2] == 'user' else 0   # source -> user = 1, shop = 0

        if self.transform is not None:
            img = self.transform(img)

        return img, label, category, source



transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5588843, 0.5189913,0.5125264), std = (0.24053435,0.24247852,0.22778076))
])




'''
labels = []
for t in train_idx:
    temp = img_paths[t].split('_')
    labels.append(int(temp[0] + temp[1].zfill(2)))

nums = {l:[0,0] for l in labels}
num_shop = {l:0 for l in labels}
num_user = {l:0 for l in labels}
for t in train_idx:
    temp = img_paths[t].split('_')
    label = int(temp[0] + temp[1].zfill(2))
    if temp[2] == 'user':
        num_user[label] += 1
        nums[label][0] += 1
    elif temp[2] == 'shop':
        num_shop[label] += 1
        nums[label][1] += 1
    else:
        print('neither 2 sources')

print(sorted(nums.items(), key=lambda x:x[1][0]))
print(sorted(Counter(num_user.values()).items(), key=lambda x:x[0]))
print(sorted(Counter(num_shop.values()).items(), key=lambda x:x[0]))
'''



'''
('source'개수: label 수)
# e.g) user image가 0개인 label수 12713개 -> positive pair 생성 못함 
# user 
[(0, 12713), (1, 9655), (2, 3058), (3, 1744), (4, 1096), (5, 725), (6, 541), 
 (7, 403), (8, 388), (9, 284), (10, 208), (11, 231), (12, 167), (13, 170), 
 (14, 128), (15, 114), (16, 110), (17, 101), (18, 108), (19, 85), (20, 70), 
 (21, 74), (22, 44), (23, 43), (24, 26), (25, 13), (26, 11), (27, 3), (28, 10), 
 (29, 4), (30, 6), (31, 4), (32, 5), (35, 1), (36, 1)]

# shop
[(0, 1302), (1, 4460), (2, 3553), (3, 3357), (4, 3483), (5, 3075), (6, 2685), 
 (7, 2147), (8, 1759), (9, 1485), (10, 1123), (11, 885), (12, 655), (13, 487), 
 (14, 318), (15, 301), (16, 220), (17, 177), (18, 152), (19, 111), (20, 89), 
 (21, 81), (22, 78), (23, 43), (24, 53), (25, 27), (26, 28), (27, 25), (28, 19), 
 (29, 26), (30, 14), (31, 13), (32, 11), (33, 12), (34, 11), (35, 7), (36, 8), 
 (37, 4), (38, 6), (39, 2), (40, 7), (41, 3), (42, 5), (43, 2), (44, 3), (45, 4), 
 (46, 2), (48, 2), (49, 3), (50, 1), (52, 2), (53, 1), (54, 1), (55, 4), (57, 1), 
 (58, 2), (59, 2), (61, 1), (63, 1), (80, 1), (82, 1), (83, 1), (132, 1), (195, 1)]

'''



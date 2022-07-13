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
from PIL import Image
from tqdm import tqdm
import os
import random



# train

# 모든 cropped 파일의 정보를 index를 추가한 dict로 list에 저장
# e.g)
#[{'idx': 0, 'pair_id': '11796', 'style': '4', 'source': 'shop', 'filename': '150388', 'item_num': 'item1', 'category_id': '11.jpg'},
# {'idx': 1, 'pair_id': '10747', 'style': '1', 'source': 'shop', 'filename': '136378', 'item_num': 'item1', 'category_id': '12.jpg'}, 
# ...
# {'idx': 9, 'pair_id': '6363', 'style': '1', 'source': 'shop', 'filename': '080672', 'item_num': 'item1', 'category_id': '4.jpg'}]

all_files = os.listdir(os.path.join('train', 'train', 'cropped'))
whole_images = []
for idx, file in tqdm(enumerate(all_files)):
    temp = file.split('_')
    whole_images.append({'idx': idx, 'pair_id':temp[0], 'style':temp[1], 'source':temp[2],
                         'filename':temp[3],'item_num':temp[4], 'category_id':temp[5]})

print(whole_images[0:10])


user_idx = []
shop_idx = []
for i in tqdm(whole_images):
    if i['source'] == 'user':
        user_idx.append(i['idx'])
    elif i['source'] == 'shop':
        shop_idx.append(i['idx'])
    else:
        print('wrong_source_included')

print('user#######',user_idx[:10])
print('shop*******',shop_idx[:10])


'''
positive_pairs_ids = []
negative_pairs_ids = []
for i in tqdm(whole_images):
    if i['source'] == 'user':
        if i['style'] != 0:
            for j in whole_images:
                if j['source'] == 'shop':
                    if i['pair_id'] == j['pair_id'] and i['style'] == j['style']:
                        positive_pairs_ids.append((i['idx'], j['idx']))
                        shop_idx.remove(j['idx'])
                        nega_sample = random.sample(shop_idx,1)      # positive item 이외의 shop idx에서 random sampling
                        negative_pairs_ids.append((i['idx'], nega_sample))

'''


#def create_positive_negative_pairs():




'''
class TripletData(Dataset):
    def __init__(self, ):




    def __getitem__(self, ):




    def __len__(self):
'''
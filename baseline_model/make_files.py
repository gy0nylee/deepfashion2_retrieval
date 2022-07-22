from tqdm import tqdm
import os
import pickle
from sklearn.model_selection import train_test_split
import numpy as np

img_paths = os.listdir(os.path.join('train', 'train', 'cropped'))

whole_images = []
for idx, path in tqdm(enumerate(img_paths)):
    temp = path.split('_')
    whole_images.append({'idx': idx, 'pair_id':temp[0], 'label':temp[0]+temp[1].zfill(2), 'source':temp[2],
                          'category':temp[5][:-4]})


pair_ids = sorted(set([i['pair_id'] for i in whole_images]))

# pair_id => 14,555개 -> 11,644(train) + 2,911(validation)
train_pair_ids, validation_pair_ids = train_test_split(pair_ids, test_size=0.2, random_state=2022)

train_idx = [i['idx'] for i in tqdm(whole_images) if i['pair_id'] in train_pair_ids]
validation_idx = [i['idx'] for i in tqdm(whole_images) if i['pair_id'] in validation_pair_ids]



# make positive_pair (anchor , positive)
# idx => img_paths에 대한 index img_paths = os.listdir(os.path.join('train', 'train', 'cropped'))
# positive pair를 담은 list 생성 -> train, validation 각각
# [(478, 102306)}, ….. (user_idx, shop_idx)]
pairs = []
for i in tqdm(np.array(whole_images)[train_idx]):
    if i['source'] == 'user':
        if i['label'][-2:] != 00:
            for j in whole_images:
                if j['source'] == 'shop':
                    if i['label'] == j['label'] :
                        pairs.append((i['idx'], j['idx']))

# [source, label, category dict], trian_idx] 생성
source = {'shop':[], 'user':[]}
label = { i['label']:[] for i in np.array(whole_images)[train_idx]}
cate = { i['category']:[] for i in np.array(whole_images)[train_idx]}

for i in tqdm(np.array(whole_images)[train_idx]):
        source[i['source']].append(i['idx'])
        label[i['label']].append(i['idx'])
        cate[i['category']].append(i['idx'])

infos_train = [pairs, source, label, cate]
with open('infos_train.pickle','wb') as f:
    pickle.dump(infos_train, f)


# test set files

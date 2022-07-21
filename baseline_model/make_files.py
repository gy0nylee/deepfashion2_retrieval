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
                          'category_id':temp[5][:-4]})


pair_ids = sorted(set([i['pair_id'] for i in whole_images]))

# pair_id => 14,555개 -> 11,644(train) + 2,911(validation)
train_pair_ids, validation_pair_ids = train_test_split(pair_ids, test_size=0.2, random_state=2022)

train_idx = [i['idx'] for i in tqdm(whole_images) if i['pair_id'] in train_pair_ids]
validation_idx = [i['idx'] for i in tqdm(whole_images) if i['pair_id'] in validation_pair_ids]

with open('train_idx.pickle','wb') as f:
    pickle.dump(train_idx, f)

with open('validation_idx.pickle','wb') as f:
    pickle.dump(validation_idx, f)



# make positive_pair (anchor , positive)
# user items의 index pool, shop items의 index pool
# -> negative sampling을 위해 생성
# idx => img_paths에 대한 index img_paths = os.listdir(os.path.join('train', 'train', 'cropped'))

# pair를 담은 list 생성
# [{p: (478, 102306), n: (478, 109536)}, {}, …..{p: (user_idx, shop_idx), n: (user_idx, shop_idx)}]
pairs = []
for i in tqdm(np.array(whole_images)[train_idx]):
    if i['source'] == 'user':
        if i['label'][-2:] != 00:
            for j in whole_images:
                if j['source'] == 'shop':
                    if i['label'] == j['label'] :
                        pairs.append((i['idx'], j['idx']))

with open('positive_pairs_train.pickle','wb') as f:
    pickle.dump(pairs, f)



for i in tqdm(whole_images):
    if i['source'] == 'user':
        user_idx.append(i['idx'])
    elif i['source'] == 'shop':
        shop_idx.append(i['idx'])
    else:
        print('wrong_source_included')





from tqdm import tqdm
import os
import random
import pickle


# train
with open('whole_images_train.pickle','rb') as f:
    whole_images_train = pickle.load(f)

# user items의 index pool, shop items의 index pool
# -> negative sampling을 위해 생성
user_idx = []    # idx => img_paths에 대한 index img_paths = os.listdir(os.path.join('train', 'train', 'cropped'))
shop_idx = []
for i in tqdm(whole_images_train):
    if i['source'] == 'user':
        user_idx.append(i['idx'])
    elif i['source'] == 'shop':
        shop_idx.append(i['idx'])
    else:
        print('wrong_source_included')


# pair를 담은 list 생성
# [{p: (478, 102306), n: (478, 109536)}, {}, …..{p: (user_idx, shop_idx), n: (user_idx, shop_idx)}]
pairs = []
for i in tqdm(whole_images_train):
    if i['source'] == 'user':
        if i['style'] != 0:
            for j in whole_images_train:
                if j['source'] == 'shop':
                    if i['pair_id'] == j['pair_id'] and i['style'] == j['style']:
                        shop_idx.remove(j['idx'])
                        neg_sample = random.sample(shop_idx,1)      # positive item 이외의 shop idx에서 random sampling
                        shop_idx.append(j['idx'])
                        pairs.append({'p':(i['idx'], j['idx']), 'n':(i['idx'], neg_sample[0])})

with open('pairs_train.pickle','wb') as f:
    pickle.dump(pairs, f)




#validation
with open('whole_images_validation.pickle','rb') as f:
    whole_images_validation = pickle.load(f)

user_idx = []
shop_idx = []
for i in tqdm(whole_images_validation):
    if i['source'] == 'user':
        user_idx.append(i['idx'])
    elif i['source'] == 'shop':
        shop_idx.append(i['idx'])
    else:
        print('wrong_source_included')

pairs = []
for i in tqdm(whole_images_validation):
    if i['source'] == 'user':
        if i['style'] != 0:
            for j in whole_images_validation:
                if j['source'] == 'shop':
                    if i['pair_id'] == j['pair_id'] and i['style'] == j['style']:
                        shop_idx.remove(j['idx'])
                        neg_sample = random.sample(shop_idx,1)      # positive item 이외의 shop idx에서 random sampling
                        shop_idx.append(j['idx'])
                        pairs.append({'p':(i['idx'], j['idx']), 'n':(i['idx'], neg_sample[0])})

with open('pairs_validation.pickle','wb') as f:
    pickle.dump(pairs, f)
with open('user_idx_val.pickle','wb') as f:
    pickle.dump(user_idx, f)
with open('shop_idx_val.pickle','wb') as f:
    pickle.dump(shop_idx, f)





# test set
with open('whole_images_test.pickle','rb') as f:
    whole_images_test = pickle.load(f)

user_idx = [] # idx => img_paths에 대한 index  img_paths = os.listdir(os.path.join('validation', 'validation', 'cropped'))
shop_idx = []
for i in tqdm(whole_images_test):
    if i['source'] == 'user':
        user_idx.append(i['idx'])
    elif i['source'] == 'shop':
        shop_idx.append(i['idx'])
    else:
        print('wrong_source_included')

pairs = []
for i in tqdm(whole_images_test):
    if i['source'] == 'user':
        if i['style'] != 0:
            for j in whole_images_test:
                if j['source'] == 'shop':
                    if i['pair_id'] == j['pair_id'] and i['style'] == j['style']:
                        shop_idx.remove(j['idx'])
                        neg_sample = random.sample(shop_idx,1)  # positive item 이외의 shop idx에서 random sampling
                        shop_idx.append(j['idx'])
                        pairs.append({'p':(i['idx'], j['idx']), 'n':(i['idx'], neg_sample[0])})

with open('pairs_test.pickle','wb') as f:
    pickle.dump(pairs, f)
with open('user_idx_test.pickle','wb') as f:
    pickle.dump(user_idx, f)
with open('shop_idx_test.pickle','wb') as f:
    pickle.dump(shop_idx, f)


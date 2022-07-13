from tqdm import tqdm
import os
import random
import pickle


# train
# 모든 cropped 파일의 정보를 index를 추가한 dict로 list에 저장
# e.g)
#[{'idx': 0, 'pair_id': '11796', 'style': '4', 'source': 'shop', 'filename': '150388', 'item_num': 'item1', 'category_id': '11'},
# {'idx': 1, 'pair_id': '10747', 'style': '1', 'source': 'shop', 'filename': '136378', 'item_num': 'item1', 'category_id': '12'},
# ...
# {'idx': 9, 'pair_id': '6363', 'style': '1', 'source': 'shop', 'filename': '080672', 'item_num': 'item1', 'category_id': '4'}]
img_paths = os.listdir(os.path.join('train', 'train', 'cropped'))
whole_images = []
for idx, path in tqdm(enumerate(img_paths)):
    temp = path.split('_')
    whole_images.append({'idx': idx, 'pair_id':temp[0], 'style':temp[1], 'source':temp[2],
                         'filename':temp[3],'item_num':temp[4], 'category_id':temp[5][:-4]})

with open('whole_images_train.pickle','wb') as f:
    pickle.dump(whole_images, f)



# user items의 index pool, shop items의 index pool
# -> negative sampling을 위해 생성
user_idx = []
shop_idx = []
for i in tqdm(whole_images):
    if i['source'] == 'user':
        user_idx.append(i['idx'])
    elif i['source'] == 'shop':
        shop_idx.append(i['idx'])
    else:
        print('wrong_source_included')


# pair를 담은 list 생성
# [{p: (478, 102306), n: (478, 109536)}, {}, …..{p: (user_idx, shop_idx), n: (user_idx, shop_idx)}]
pairs = []
for i in tqdm(whole_images):
    if i['source'] == 'user':
        if i['style'] != 0:
            for j in whole_images:
                if j['source'] == 'shop':
                    if i['pair_id'] == j['pair_id'] and i['style'] == j['style']:
                        shop_idx.remove(j['idx'])
                        neg_sample = random.sample(shop_idx,1)      # positive item 이외의 shop idx에서 random sampling
                        shop_idx.append(j['idx'])
                        pairs.append({'p':(i['idx'], j['idx']), 'n':(i['idx'], neg_sample[0])})

with open('pairs_train.pickle','wb') as f:
    pickle.dump(pairs, f)





# validation
img_paths = os.listdir(os.path.join('validation', 'validation', 'cropped'))
whole_images = []
for idx, path in tqdm(enumerate(img_paths)):
    temp = path.split('_')
    whole_images.append({'idx': idx, 'pair_id':temp[0], 'style':temp[1], 'source':temp[2],
                         'filename':temp[3],'item_num':temp[4], 'category_id':temp[5][:-4]})

with open('whole_images_validation.pickle','wb') as f:
    pickle.dump(whole_images, f)


# user items의 index pool, shop items의 index pool
user_idx = []
shop_idx = []
for i in tqdm(whole_images):
    if i['source'] == 'user':
        user_idx.append(i['idx'])
    elif i['source'] == 'shop':
        shop_idx.append(i['idx'])
    else:
        print('wrong_source_included')


# pair를 담은 list 생성
pairs = []
for i in tqdm(whole_images):
    if i['source'] == 'user':
        if i['style'] != 0:
            for j in whole_images:
                if j['source'] == 'shop':
                    if i['pair_id'] == j['pair_id'] and i['style'] == j['style']:
                        shop_idx.remove(j['idx'])
                        neg_sample = random.sample(shop_idx,1)      # positive item 이외의 shop idx에서 random sampling
                        shop_idx.append(j['idx'])
                        pairs.append({'p':(i['idx'], j['idx']), 'n':(i['idx'], neg_sample[0])})

with open('pairs_validation.pickle','wb') as f:
    pickle.dump(pairs, f)




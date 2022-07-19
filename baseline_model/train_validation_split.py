from tqdm import tqdm
import os
import pickle

# train, validation
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


pair_ids = list(set([i['pair_id'] for i in whole_images]))
# pair_id => 14,555개 -> 11,644(train) + 2,911(test)
train_pair_ids = pair_ids[:11644]
validation_pair_ids = pair_ids[11644:]



whole_images_train = []
whole_images_validation = []
for i in tqdm(whole_images):
    if i['pair_id'] in train_pair_ids:
        whole_images_train.append(i)
    elif i['pair_id'] in validation_pair_ids:
        whole_images_validation.append(i)
    else:
        print('not in any set')

with open('whole_images_train.pickle','wb') as f:
    pickle.dump(whole_images_train, f)
with open('whole_images_validation.pickle','wb') as f:
    pickle.dump(whole_images_validation, f)



# test
img_paths = os.listdir(os.path.join('validation', 'validation', 'cropped'))
whole_images_test = []
for idx, path in tqdm(enumerate(img_paths)):
    temp = path.split('_')
    whole_images_test.append({'idx': idx, 'pair_id':temp[0], 'style':temp[1], 'source':temp[2],
                         'filename':temp[3],'item_num':temp[4], 'category_id':temp[5][:-4]})

with open('whole_images_test.pickle','wb') as f:
    pickle.dump(whole_images_test, f)
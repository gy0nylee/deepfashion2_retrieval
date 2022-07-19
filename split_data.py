from tqdm import tqdm
import os
import pickle
from sklearn.model_selection import train_test_split

img_paths = os.listdir(os.path.join('train', 'train', 'cropped'))

whole_pair_ids = []
category = []
for idx, path in tqdm(enumerate(img_paths)):
    temp = path.split('_')
    whole_pair_ids.append({'idx': idx, 'pair_id': temp[0]})
    category.append(temp[5][:-4])

pair_ids = sorted(set([i['pair_id'] for i in whole_pair_ids]))

# pair_id => 14,555개 -> 11,644(train) + 2,911(validation)
train_pair_ids, validation_pair_ids = train_test_split(pair_ids, test_size=0.2, random_state=2022)

train_idx = [i['idx'] for i in tqdm(whole_pair_ids) if i['pair_id'] in train_pair_ids]
validation_idx = [i['idx'] for i in tqdm(whole_pair_ids) if i['pair_id'] in validation_pair_ids]

with open('train_idx.pickle','wb') as f:
    pickle.dump(train_idx, f)

with open('validation_idx.pickle','wb') as f:
    pickle.dump(validation_idx, f)






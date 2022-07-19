from tqdm import tqdm
import os
import pickle

img_paths = os.listdir(os.path.join('train', 'train', 'cropped'))

for idx, path in tqdm(enumerate(img_paths)):
    temp = path.split('_')



pair_ids = list(set([i['pair_id'] for i in whole_images]))
# pair_id => 14,555개 -> 11,644(train) + 2,911(test)
train_pair_ids = pair_ids[:11644]
validation_pair_ids = pair_ids[11644:]

train_idx = [t['idx'] for t in tqdm(whole_images_train)]

import json
import os
from collections import Counter

# Train
json_name = os.path.join('deepfashion2_coco_' + 'train' + '.json')
with open(json_name, 'r') as f:
    train_js = json.loads(f.read())



# image 개수 : => # of images: 191961
print('# of images:', len(train_js['images']))
# image 크기 분포
train_images = {"width": [], "height": []}
for image in train_js['images']:
    train_images['width'].append(image['width'])
    train_images['height'].append(image['height'])



# annotation 개수
# ...


# pair 개수 : annotations 'pair_id'
# user, shop 개수 => annotations 'source'
# 카테고리 분포 'category_id'
# style 분포 'style'
train_anno = {"pair_id": [], "source": [], "category_id": [], "style": []}
for anno in train_js['annotations']:
    train_anno['pair_id'].append(anno['pair_id'])
    train_anno['source'].append(anno['source'])
    train_anno['category_id'].append(anno['category_id'])
    train_anno['style'].append(anno['style'])

source_count = Counter(train_anno['source'])
category_count = Counter(train_anno['category_id'])
style_count = Counter(train_anno['style'])

print(len(set(train_anno['pair_id'])))
print(source_count)
print(category_count)
print(style_count)

# pair 개수 : 14555
# source : Counter({'shop': 228558, 'user': 83628})
# category : Counter({1: 71645, 8: 55387, 7: 36616, 2: 36064, 9: 30835, 12: 17949, 10: 17211, 5: 16095, 4: 13457, 11: 7907, 13: 6492, 6: 1985, 3: 543})
# style : Counter({1: 130185, 0: 94408, 2: 57035, 3: 16032, 4: 7580, 5: 2983, 6: 1844, 7: 722, 8: 507, 9: 292, 10: 191, 11: 100, 12: 85, 13: 51, 14: 51, 16: 29, 15: 24, 18: 15, 17: 14, 19: 7, 27: 4, 21: 4, 20: 4, 23: 4, 31: 2, 30: 2, 26: 2, 33: 2, 22: 2, 32: 1, 28: 1, 25: 1, 29: 1, 24: 1})





# Validation
json_name = os.path.join('deepfashion2_coco_' + 'validation' + '.json')
with open(json_name, 'r') as f:
    val_js = json.loads(f.read())


#print(val_js.keys())
#print('images', val_js['images']) # width, height
#print('annos', val_js['annotations'])
#print(val_js['categories']) #category



# image 개수 => # of images: 32153
print('# of images:', len(val_js['images']))
# image 크기 분포
val_images = {"width": [], "height": []}
for image in val_js['images']:
    val_images['width'].append(image['width'])
    val_images['height'].append(image['height'])
# width, height 분포 그리기



# annotation 개수
# ...


# pair 개수 : annotations 'pair_id'
# user, shop 개수 => annotations 'source'
# 카테고리 분포 'category_id'
# style 분포 'style'
val_anno = {"pair_id": [], "source": [], "category_id": [], "style": []}
for anno in val_js['annotations']:
    val_anno['pair_id'].append(anno['pair_id'])
    val_anno['source'].append(anno['source'])
    val_anno['category_id'].append(anno['category_id'])
    val_anno['style'].append(anno['style'])

source_count = Counter(val_anno['source'])
category_count = Counter(val_anno['category_id'])
style_count = Counter(val_anno['style'])

print(len(set(val_anno['pair_id'])))
print(source_count)
print(category_count)
print(style_count)

# pair 개수 :2279
# source : Counter({'shop': 36961, 'user': 15529})
# category : Counter({1: 12556, 8: 9586, 9: 6522, 2: 5966, 7: 4167, 12: 3352, 10: 3127, 5: 2113, 4: 2011, 11: 1477, 13: 1149, 6: 322, 3: 142})
# style : Counter({1: 21111, 0: 16240, 2: 9606, 3: 2894, 4: 1402, 5: 476, 6: 271, 7: 156, 8: 116, 9: 63, 10: 53, 11: 39, 12: 18, 13: 16, 14: 9, 15: 7, 16: 4, 17: 3, 23: 1, 18: 1, 21: 1, 20: 1, 22: 1, 19: 1})
# category_id - lst_name 연결
# style의 의미...?




# test -> image만 존재

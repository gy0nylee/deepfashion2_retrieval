import json
import os
from collections import Counter
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Train
#json_name = os.path.join('deepfashion2_coco_' + 'train' + '.json')
#with open(json_name, 'r') as f:
#    train_js = json.loads(f.read())



# image 개수 : => # of images: 191961
#print('# of images:', len(train_js['images']))
# image 크기 분포
#train_images = {"width": [], "height": []}
#for image in train_js['images']:
#    train_images['width'].append(image['width'])
#    train_images['height'].append(image['height'])



# annotation 개수
# ...


# pair 개수 : annotations 'pair_id'
# user, shop 개수 => annotations 'source'
# 카테고리 분포 'category_id'
# style 분포 'style'
#train_anno = {"pair_id": [], "source": [], "category_id": [], "style": []}
#for anno in train_js['annotations']:
#    train_anno['pair_id'].append(anno['pair_id'])
#    train_anno['source'].append(anno['source'])
#    train_anno['category_id'].append(anno['category_id'])
#   train_anno['style'].append(anno['style'])

#source_count = Counter(train_anno['source'])
#category_count = Counter(train_anno['category_id'])
#style_count = Counter(train_anno['style'])

#print(len(set(train_anno['pair_id'])))
#print(source_count)
#print(category_count)
#print(style_count)

# pair 개수 : 14555
# source : Counter({'shop': 228558, 'user': 83628})
# category : Counter({1: 71645, 8: 55387, 7: 36616, 2: 36064, 9: 30835, 12: 17949, 10: 17211, 5: 16095, 4: 13457, 11: 7907, 13: 6492, 6: 1985, 3: 543})
# style : Counter({1: 130185, 0: 94408, 2: 57035, 3: 16032, 4: 7580, 5: 2983, 6: 1844, 7: 722, 8: 507, 9: 292, 10: 191, 11: 100, 12: 85, 13: 51, 14: 51, 16: 29, 15: 24, 18: 15, 17: 14, 19: 7, 27: 4, 21: 4, 20: 4, 23: 4, 31: 2, 30: 2, 26: 2, 33: 2, 22: 2, 32: 1, 28: 1, 25: 1, 29: 1, 24: 1})





# Validation
all_files = os.listdir(os.path.join('validation', 'validation', 'image'))
val = []
item_count = []
for file in tqdm(all_files):
    count = 0
    json_name = os.path.join('validation', 'validation', 'annos', os.path.splitext(file)[0] + '.json')
    image_name = os.path.join('validation', 'validation', 'image', file)
    if int(os.path.splitext(file)[0]) >= 0:
        img = Image.open(image_name)
        width, height = img.size
        with open(json_name, 'r') as f:
            anno = json.loads(f.read())
            val.append(anno)
            for i in anno:
                if i == 'source' or i == 'pair_id':
                    continue
                else:
                    count += 1
            item_count.append(count)

print(Counter(item_count)) # Counter({2: 18787, 1: 12684, 3: 498, 4: 182, 5: 2})




json_name = os.path.join('deepfashion2_coco_' + 'validation' + '.json')
with open(json_name, 'r') as f:
    val_js = json.loads(f.read())


#val1013 = os.path.join('validation', 'validation', 'annos', '014746' + '.json')
#with open(val1013, 'r') as f:
#    val_1013_js = json.loads(f.read())




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

plt.hist(val_images['width'], bins=50)
plt.xticks(np.arange(0, 1000, 50), rotation=90)
plt.xlim(0, 1000)
plt.title('Image_width')
plt.savefig('width.png')
plt.show()


plt.hist(val_images['height'], bins=50)
plt.xticks(np.arange(0, 1000, 50), rotation=90)
plt.xlim(0, 1000)
plt.title('Image_height')
plt.savefig('height.png')
plt.show()


#plt.xticks(np.arange(1, 8, 1))
#plt.yticks(np.arange(0, 1.1, 0.1))
#plt.xlabel('')
#plt.ylabel('')
#plt.legend()
#plt.title('')

# annotation 개수
#


# pair 개수 : annotations 'pair_id'
# user, shop 개수 => annotations 'source'
# 카테고리 분포 'category_id'
# style 분포 'style'
# style => pair에 존재하는 style 개수?


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

#D = {u'Label1':26, u'Label2': 17, u'Label3':30}

#plt.bar(range(len(D)), list(D.values()), align='center')
#plt.xticks(range(len(D)), list(D.keys()))



# category_id - lst_name 연결
lst_name = ['short_sleeved_shirt', 'long_sleeved_shirt', 'short_sleeved_outwear', 'long_sleeved_outwear',
            'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short_sleeved_dress',
            'long_sleeved_dress', 'vest_dress', 'sling_dress']

cate_dict = {lst_name[i-1]:count for i,count in category_count.items()}




# test -> image만 존재
#all_files = os.listdir(os.path.join('test', 'test', 'image'))
#test_ds = []
#for idx, file in tqdm(enumerate(all_files)):
#    image_name = os.path.join('test', 'test', 'image', file)
 #   if int(os.path.splitext(file)[0]) >= 0:
 #       img = Image.open(image_name)
 #       width, height = img.size
 #       test_ds.append({'idx':idx, 'image':img, 'size':(width,height)})

#print(test_ds[0])




#import cv2
import matplotlib.pyplot as plt
import json
import os
from PIL import Image
from tqdm import tqdm


'''
image_name = os.path.join('validation', 'validation', 'image', '000001.jpg')
img = Image.open(image_name)
width, height = img.size

val01 = os.path.join('validation', 'validation', 'annos', '000001' + '.json')
with open(val01, 'r') as f:
    val_01_js = json.loads(f.read())
pair_id = val_01_js['pair_id']
a,b,c,d = val_01_js['item1']['bounding_box']
print(a,b,c,d)
img2 = img.crop((a,b,c,d))
i = 'item1'
img2.save(os.path.join('validation', 'validation','cropped', str(pair_id)+'_'+str(val_01_js[i]['style'])+'_'+'000001'+'_'+f'{i}'+'.jpg'), 'JPEG')
'''

#  pair_id_style_파일명_item명.jpg
#  validation

all_files = os.listdir(os.path.join('validation', 'validation', 'image'))
for file in tqdm(all_files):
    count = 0
    json_name = os.path.join('validation', 'validation', 'annos', os.path.splitext(file)[0] + '.json')
    image_name = os.path.join('validation', 'validation', 'image', file)
    if int(os.path.splitext(file)[0]) >= 0:
        img = Image.open(image_name)
        width, height = img.size
        with open(json_name, 'r') as f:
            anno = json.loads(f.read())
            pair_id = anno['pair_id']
            for i in anno:
                if i == 'source' or i == 'pair_id':
                    continue
                else:
                    a,b,c,d = anno[i]['bounding_box']
                    img2 = img.crop((a,b,c,d))
                    img2.save(os.path.join('validation', 'validation', 'cropped', str(pair_id)+'_'+str(anno[i]['style'])+'_'+os.path.splitext(file)[0]+'_'+f'{i}'+'.jpg'), 'JPEG')


#  train

all_files = os.listdir(os.path.join('train', 'train', 'image'))
for file in tqdm(all_files):
    count = 0
    json_name = os.path.join('train', 'train', 'annos', os.path.splitext(file)[0] + '.json')
    image_name = os.path.join('train', 'train', 'image', file)
    if int(os.path.splitext(file)[0]) >= 0:
        img = Image.open(image_name)
        width, height = img.size
        with open(json_name, 'r') as f:
            anno = json.loads(f.read())
            pair_id = anno['pair_id']
            for i in anno:
                if i == 'source' or i == 'pair_id':
                    continue
                else:
                    a, b, c, d = anno[i]['bounding_box']
                    img2 = img.crop((a, b, c, d))
                    img2.save(os.path.join('train', 'train', 'cropped',
                                           str(pair_id) + '_' + str(anno[i]['style']) + '_' + os.path.splitext(file)[
                                               0] + '_' + f'{i}' + '.jpg'), 'JPEG')


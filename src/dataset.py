import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import os, json
import numpy as np
from PIL import Image
import pickle
from tqdm import tqdm
from collections import defaultdict

def collate_fn(batch):
    items = list(zip(*batch))
    items[0] = default_collate([i for i in items[0] if torch.is_tensor(i)])
    items[1] = list([i for i in items[1] if i])
    items[2] = list([i for i in items[2] if i])
    items[3] = default_collate([i for i in items[3] if torch.is_tensor(i)])
    items[4] = default_collate([i for i in items[4] if torch.is_tensor(i)])
    return items

# [참가자 TO-DO] custom dataset
class Small_dataset(Dataset):
    def __init__(self, label_data, num_classes=80, transform=None):
        self.label_data = label_data
        self.num_class = num_classes + 1 # 배경 포함 모델
        self.transform = transform

        self.img_list = self.label_data['data']
        self.label_list = self.label_data['label']

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        image = self.img_list[index].copy()
        label = self.label_list[index]

        height, width = label[0]
        labels = label[1].clone()
        boxes = label[2].clone()

        # [참가자 TO-DO] 모델에 맞는 전처리 코드로 채우면 됩니다. 
        if self.transform is not None:
            image, (height, width), boxes, labels =\
                self.transform(image, (height, width), boxes, labels)
        return image, int(index), (height, width), boxes, labels

# [참가자 TO-DO] 효율적인 훈련 시간을 위한 preprocessing / 사용하지 않아도 무방합니다.
def prepocessing(root_dir, label_data, input_size):
    img_file_list = list(label_data.keys())
    img_list = []
    label_list = []

    print("Starting Caching...")
    for idx, cur_file in enumerate(tqdm(img_file_list)):
        image = Image.open(os.path.join(root_dir, cur_file))
        width, height = image.size
        img_list.append(image.resize(input_size))

        # 원본 레이블 형식 list [cls, x, y, w, h]
        cur_label = np.array(label_data[cur_file])

        boxes = []
        labels = []
        
        for label in cur_label:
            # 네트워크에 따라 cls index를 1씩(background) 미룹니다.
            labels.append(int(label[0]+1))

            # model 형식에 맞게 변환 필요 (ex. xywh -> (normalized left,top,right,bottom)
            boxes.append([label[1] / width, label[2] / height, 
                (label[1] + label[3]) / width, (label[2]+label[4]) / height])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels)

        label_list.append(((height, width), labels, boxes))
    
    data_dict = {
        'data':img_list,
        'label':label_list  
    }

    return data_dict


coco_dict = dict(
    info= dict(
        year=2021, 
        version="1.0", 
        description="obd", 
        contributor="AI", 
        url=None, 
        date_created=None
    ), # year, version, description, contributor, url, date_created
    licenses=dict(
        id=0,
        name='CC BY 4.0',
        url=''
    ), # id, name, url
    images=list(), # id, file_name, height, width
    annotations=list(), # id, image_id, category_id, bbox, area
    categories=list() # id, name, supercategory
)

def convert_to_coco_train(json_path, classes, coco_dict):
    classes_count = {key:value for key, value in zip(range(30), [0]*30)}

    train_indices = list(map(lambda x:str(x), range(17173)[17173//5:]))

    img = []
    annotations = []
    categories = []

    img_idx = 0
    anno_idx = 0
    cat_idx = 0

    for c in classes:
        categories.append({'id':cat_idx, 'name':c, 'super_category':None})
        cat_idx += 1

    with open(json_path, 'r') as jfile:
        
        json_data = json.load(jfile) # dict

        for key in json_data:
            if key.split('.')[0] in train_indices:
                img.append({'id':img_idx, 'file_name': key, 'height':2100, 'width':2800})
            
                for label, x, y, w, h in json_data[key]:
                    label,x,y,w,h = int(label), int(x), int(y), int(w), int(h)
                    annotations.append({'id':anno_idx, 'image_id':img_idx, 'category_id':label, 'bbox':(x,y,w,h), 'area':w*h, 'iscrowd':0,\
                                    'ignore':0, 'segmentation': []})
                    classes_count[int(label)] += 1
                    anno_idx += 1
                img_idx += 1
            
    coco_dict['images'] = img
    coco_dict['annotations'] = annotations
    coco_dict['categories'] = categories

    CUR_PATH = os.getcwd()
    with open(os.path.join(CUR_PATH, 'all_train.json'), 'w', encoding='utf-8') as jfile:
        json.dump(coco_dict, jfile)
  
    print(classes_count)

def convert_to_coco_valid(json_path, classes, coco_dict):
    classes_count = {key:value for key, value in zip(range(30), [0]*30)}
    
    valid_indices = list(map(lambda x:str(x), range(17173)[:17173//5]))

    img = []
    annotations = []
    categories = []

    img_idx = 0
    anno_idx = 0
    cat_idx = 0

    for c in classes:
        categories.append({'id':cat_idx, 'name':c, 'super_category':None})
        cat_idx += 1

    with open(json_path, 'r') as jfile:
        json_data = json.load(jfile) # dict

        for key in json_data:
            if key.split('.')[0] in valid_indices:
                img.append({'id':img_idx, 'file_name': key, 'height':2100, 'width':2800})
            
                for label, x, y, w, h in json_data[key]:
                    label,x,y,w,h = int(label), int(x), int(y), int(w), int(h)
                    annotations.append({'id':anno_idx, 'image_id':img_idx, 'category_id':label, 'bbox':(x,y,w,h),\
                        'area':w*h, 'iscrowd':0, 'ignore':0, 'segmentation': []})

                    classes_count[int(label)] += 1
                    anno_idx += 1
                img_idx += 1
            
    coco_dict['images'] = img
    coco_dict['annotations'] = annotations
    coco_dict['categories'] = categories

    CUR_PATH = os.getcwd()
    with open(os.path.join(CUR_PATH, 'valid.json'), 'w', encoding='utf-8') as jfile:
        json.dump(coco_dict, jfile)

def convert_to_coco_test(img_paths, classes, coco_dict):
    img = []
    annotations = []
    categories = []

    img_idx = 0
    anno_idx = 0
    cat_idx = 0

    for c in classes:
        categories.append({'id':cat_idx, 'name':c, 'super_category':None})
        cat_idx += 1
    
    for img_path in img_paths:
        file_name = img_path.split('/')[-1]
        img.append({'id':img_idx, 'file_name': file_name, 'height':2100, 'width':2800})
    
    for _ in range(len(img_paths)):
        label,x,y,w,h = 0, 0, 0, 1, 1
        annotations.append({'id':anno_idx, 'image_id':img_idx, 'category_id':label, 'bbox':(x,y,w,h),\
            'area':w*h, 'iscrowd':0, 'ignore':0, 'segmentation': []})

        anno_idx += 1
    img_idx += 1

    coco_dict['images'] = img
    coco_dict['annotations'] = annotations
    coco_dict['categories'] = categories

    CUR_PATH = os.getcwd()
    with open(os.path.join(CUR_PATH, 'test.json'), 'w', encoding='utf-8') as jfile:
        json.dump(coco_dict, jfile)

if __name__ == '__main__':
    coco_dict = dict(
        info= dict(
            year=2021, 
            version="1.0", 
            description="obd", 
            contributor="AI", 
            url=None, 
            date_created=None
        ), # year, version, description, contributor, url, date_created
        licenses=dict(
            id=0,
            name='CC BY 4.0',
            url=''
        ), # id, name, url
        images=list(), # id, file_name, height, width
        annotations=list(), # id, image_id, category_id, bbox, area
        categories = list() # id, name, supercategory
    )
    classes = ['SD카드', '웹캠', 'OTP', '계산기', '목걸이', '넥타이핀', '십원', '오십원', '백원', '오백원', '미국지폐', '유로지폐', '태국지폐', '필리핀지폐',
            '밤', '브라질너트', '은행', '피칸', '호두', '호박씨', '해바라기씨', '줄자', '건전지', '망치', '못', '나사못', '볼트', '너트', '타카', '베어링']
    convert_to_coco_valid(r'C:\Users\JCdata\workspace\small_obd\test\test.json', classes, coco_dict)
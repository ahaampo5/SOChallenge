import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import os
import numpy as np
import cv2
import pickle
from tqdm import tqdm

def collate_fn(batch):
    return tuple(zip(*batch))

# [참가자 TO-DO] custom dataset
class Small_dataset(Dataset):
    def __init__(self, label_data, transform=None):
        self.label_data = label_data
        self.transforms = transform

        self.img_list = self.label_data['data']
        self.label_list = self.label_data['label']

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        #image = self.img_list[index].copy()
        image = cv2.imread(self.img_list[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        label = self.label_list[index]

        t_labels = label[0].clone()
        t_boxes = label[1].clone()

        # [참가자 TO-DO] 모델에 맞는 전처리 코드로 채우면 됩니다. 
        if self.transforms:
            sample = {
                'image': image,
                'bboxes': t_boxes,
                'labels': t_labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            target = {
                'boxes':torch.tensor(sample['bboxes'], dtype=torch.float32),
                'labels':torch.tensor(sample['labels'], dtype=torch.int64)
            }

        return image, target

# [참가자 TO-DO] 효율적인 훈련 시간을 위한 preprocessing / 사용하지 않아도 무방합니다.
def prepocessing(root_dir, label_data, input_size):
    img_file_list = list(label_data.keys())
    img_list = []
    label_list = []

    for idx, cur_file in enumerate(img_file_list):
        '''
        image = cv2.imread(os.path.join(root_dir, cur_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = cv2.resize(image, dsize=input_size)
        
        image /= 255.0
        '''
        image = os.path.join(root_dir, cur_file)

        img_list.append(image)

        # 원본 레이블 형식 list [cls, x, y, w, h]
        cur_label = np.array(label_data[cur_file])

        boxes = []
        labels = []
        
        for label in cur_label:
            # 네트워크에 따라 cls index를 1씩(background) 미룹니다.
            if (label[3]==0 or label[4]==0):
                print("wrong box")
                continue
            else:
                labels.append(int(label[0]+1))
                # model 형식에 맞게 변환 필요 (ex. xywh -> (normalized left,top,right,bottom)
                # modify : xywh -> xyxy
                boxes.append([label[1], label[2], (label[1] + label[3]), (label[2]+label[4])])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels)

        label_list.append((labels, boxes))
    
    data_dict = {
        'data':img_list,
        'label':label_list
    }
    return data_dict
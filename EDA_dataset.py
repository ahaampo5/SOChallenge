import os
import json
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader

# baseline model
from src.model import BASELINE_MODEL
from src.utils import train, generate_dboxes, Encoder, BaseTransform
from src.loss import Loss
from src.dataset import collate_fn, Small_dataset, prepocessing,\
    coco_dict, convert_to_coco_train, convert_to_coco_valid

# nsml
import nsml
from nsml import DATASET_PATH

import sys
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector, set_random_seed, init_detector
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.utils import collect_env, get_root_logger



# only infer
def test_preprocessing(img, transform=None):
    # [참가자 TO-DO] inference를 위한 이미지 데이터 전처리
    if transform is not None:
        img = transform(img)
        img = img.unsqueeze(0)
    return img

def bind_model(model):
    def save(dir_path, **kwargs):
        checkpoint = {
            "model": model.state_dict()}
        torch.save(checkpoint, os.path.join(dir_path, 'model.pt'))
        print("model saved!")

    def load(dir_path):
        checkpoint = torch.load(os.path.join(dir_path, 'model.pt'))
        model.load_state_dict(checkpoint["model"])
        print('model loaded!')

    def infer(test_img_path_list): # data_loader에서 인자 받음
        '''
        반환 형식 준수해야 정상적으로 score가 기록됩니다.
        {'file_name':[[cls_num, x, y, w, h, conf]]}
        '''
        result_dict = {}

        # for baseline model ==============================
        import torchvision.transforms as transforms
        from PIL import Image
        from tqdm import tqdm

        infer_transforms = transforms.Compose([
                transforms.Resize((300,300)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        dboxes = generate_dboxes() 
        encoder = Encoder(dboxes) # inference시 박스 좌표로 후처리하는 모듈

        model.cuda()
        model.eval()

        for _, file_path in enumerate(tqdm(test_img_path_list)):
            file_name = file_path.split("/")[-1]
            img = Image.open(file_path)
            width, height = img.size

            img = test_preprocessing(img, infer_transforms)
            img = img.cuda()
            detections = []

            with torch.no_grad():
                ploc, plabel = model(img)
                ploc, plabel = ploc.float().detach().cpu(), plabel.float().detach().cpu()

                try:
                    result = encoder.decode_batch(ploc, plabel, 0.5, 100)[0]
                except:
                    print("No object detected : ", file_name)
                    continue

                loc, label, prob = [r.numpy() for r in result]
                for loc_, label_, prob_ in zip(loc, label, prob):
                    try:
                        '''
                        결과 기록 형식, 데이터 타입 준수해야 함
                        pred_cls, x, y, w, h, confidence
                        '''
                        detections.append([
                            int(label_)-1,
                            float( loc_[0] * width ), 
                            float( loc_[1] * height ), 
                            float( (loc_[2] - loc_[0]) * width ),
                            float( (loc_[3] - loc_[1]) * height ), 
                            float( prob_ )
                            ])
                    except:
                        continue

            result_dict[file_name] = detections # 반환 형식 준수해야 함
        return result_dict

    # DONOTCHANGE: They are reserved for nsml
    nsml.bind(save=save, load=load, infer=infer)

def get_args():
    parser = ArgumentParser(description="NSML BASELINE")
    parser.add_argument("--epochs", type=int, default=10, help="number of total epochs to run")
    parser.add_argument("--batch-size", type=int, default=8, help="number of samples for each iteration")
    parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate")
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--use_mmdet", type=bool, default=False)

    # DONOTCHANGE: They are reserved for nsml
    parser.add_argument("--pause", type=int, default=0)
    parser.add_argument('--mode', type=str, default='train', help='submit일때 test로 설정됩니다.')
    parser.add_argument('--iteration', type=str, default='0', help='fork 명령어를 입력할때의 체크포인트로 설정됩니다.')    
    args = parser.parse_args()
    return args

def main(opt):
    num_class = 30
    if opt.use_mmdet:
        classes = ['SD카드', '웹캠', 'OTP', '계산기', '목걸이', '넥타이핀', '십원', '오십원', '백원', '오백원', '미국지폐', '유로지폐', '태국지폐', '필리핀지폐',
            '밤', '브라질너트', '은행', '피칸', '호두', '호박씨', '해바라기씨', '줄자', '건전지', '망치', '못', '나사못', '볼트', '너트', '타카', '베어링']
            
        _, result = convert_to_coco_train(os.path.join(DATASET_PATH, 'train', 'train_label'),
            classes, coco_dict
        )
    
        width = []
        height = []
        labels =  [] 
        annos = result['annotations']

        for ann in annos:
            width.append(ann["bbox"][2])
            height.append(ann["bbox"][3])
            labels.append(ann["category_id"])

        print("[[width]]\n", width)
        print("[[height]]\n", height)
        print("[[label]]\n", labels)
    else:
        dboxes = generate_dboxes()

        with open(os.path.join(DATASET_PATH, 'train', 'train_label'), 'r', encoding="utf-8") as f:
            train_data_dict = json.load(f)
            train_img_label = prepocessing(root_dir=os.path.join(DATASET_PATH, 'train', 'train_data'),\
                label_data=train_data_dict, input_size=(300,300))
        
        # data loader
        train_data = Small_dataset(train_img_label, num_class, BaseTransform(dboxes))
        
        for img, idx, (height, width), gloc, glabel in train_data:
            if type(height) == int or type(height) == float:
                if h == 0 or w == 0:
                    print(f"{idx} img has zero box")
            else:
                for h, w in zip(height, width):
                    if h == 0 or w == 0:
                        print(f"{idx} img has zero box")

if __name__ == "__main__":
    opt = get_args()
    main(opt)

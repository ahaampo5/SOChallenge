import os
import json
<<<<<<< HEAD
import numpy as np
=======
>>>>>>> dbd4d62f0278ccf3c5c041c285abf3c064bf5f9a
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader

# baseline model
<<<<<<< HEAD
from src.model import build_model
from src.utils import train
from src.dataset import collate_fn, Small_dataset, prepocessing

# albumentation
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

=======
from src.model import BASELINE_MODEL
from src.utils import train, generate_dboxes, Encoder, BaseTransform
from src.loss import Loss
from src.dataset import collate_fn, Small_dataset, prepocessing

>>>>>>> dbd4d62f0278ccf3c5c041c285abf3c064bf5f9a
# nsml
import nsml
from nsml import DATASET_PATH

<<<<<<< HEAD
# multi-gpu
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

# wbf
from ensemble_boxes import *

# IMAGE SIZE
IMAGE_SIZE = 1024

=======
>>>>>>> dbd4d62f0278ccf3c5c041c285abf3c064bf5f9a
# only infer
def test_preprocessing(img, transform=None):
    # [참가자 TO-DO] inference를 위한 이미지 데이터 전처리
    if transform is not None:
<<<<<<< HEAD
        img = transform(image=img)['image']
=======
        img = transform(img)
>>>>>>> dbd4d62f0278ccf3c5c041c285abf3c064bf5f9a
        img = img.unsqueeze(0)
    return img

def bind_model(model):
    def save(dir_path, **kwargs):
<<<<<<< HEAD
        torch.save(model.module.state_dict(), os.path.join(dir_path, 'model.pt'))
=======
        checkpoint = {
            "model": model.state_dict()}
        torch.save(checkpoint, os.path.join(dir_path, 'model.pt'))
>>>>>>> dbd4d62f0278ccf3c5c041c285abf3c064bf5f9a
        print("model saved!")

    def load(dir_path):
        checkpoint = torch.load(os.path.join(dir_path, 'model.pt'))
<<<<<<< HEAD
        model.load_state_dict(checkpoint)
        print('model loaded!')

    def get_test_transform():
        return A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            ToTensorV2(p=1.0)
        ])

    def run_wbf(pred, iou_thr=0.5, skip_box_thr=0.05, weights=None):
        boxes = (pred['boxes']/1024.).tolist()
        scores = pred['scores'].tolist()
        labels = pred['labels'].tolist()
        boxes, scores, labels = weighted_boxes_fusion([boxes], [scores], [labels], weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        return boxes, scores, labels

=======
        model.load_state_dict(checkpoint["model"])
        print('model loaded!')

>>>>>>> dbd4d62f0278ccf3c5c041c285abf3c064bf5f9a
    def infer(test_img_path_list): # data_loader에서 인자 받음
        '''
        반환 형식 준수해야 정상적으로 score가 기록됩니다.
        {'file_name':[[cls_num, x, y, w, h, conf]]}
        '''
        result_dict = {}

        # for baseline model ==============================
<<<<<<< HEAD
        from tqdm import tqdm

=======
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
        
>>>>>>> dbd4d62f0278ccf3c5c041c285abf3c064bf5f9a
        model.cuda()
        model.eval()

        for _, file_path in enumerate(tqdm(test_img_path_list)):
            file_name = file_path.split("/")[-1]
<<<<<<< HEAD
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            #img /= 255.0

            width = img.shape[1]
            height = img.shape[0]

            img = test_preprocessing(img, get_test_transform())
            img = img.cuda()
            detections = []
            count = 0

            with torch.no_grad():
                pred = model(img)[0]
                try:
                    boxes, scores, labels = run_wbf(pred, iou_thr=0.5, skip_box_thr=0.05)
                except:
                    continue

                '''
                wbf_pred = []
                wbf_pred.append(boxes)
                wbf_pred.append(scores)
                wbf_pred.append(labels)

                wbf_pred = np.array(wbf_pred)
                wbf_pred = np.transpose(wbf_pred)
                wbf_pred = sorted(wbf_pred, key=lambda x:x[1], reverse=True)
                wbf_pred = np.transpose(wbf_pred)
                wbf_pred = [list(t) for t in wbf_pred]
                boxes, scores, labels = wbf_pred
                '''
                for box_, score_, label_ in zip(boxes, scores, labels):
                    try:
                        detections.append([
                            int(label_)-1,
                            float( box_[0] * width ),
                            float( box_[1] * height ), 
                            float( (box_[2] - box_[0]) * width ),
                            float( (box_[3] - box_[1]) * height ), 
                            float( score_ )
=======
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
>>>>>>> dbd4d62f0278ccf3c5c041c285abf3c064bf5f9a
                            ])
                    except:
                        continue

            result_dict[file_name] = detections # 반환 형식 준수해야 함
        return result_dict

    # DONOTCHANGE: They are reserved for nsml
    nsml.bind(save=save, load=load, infer=infer)

def get_args():
    parser = ArgumentParser(description="NSML BASELINE")
<<<<<<< HEAD
    parser.add_argument("--epochs", type=int, default=20, help="number of total epochs to run")
    parser.add_argument("--batch-size", type=int, default=4, help="number of samples for each iteration")
=======
    parser.add_argument("--epochs", type=int, default=10, help="number of total epochs to run")
    parser.add_argument("--batch-size", type=int, default=8, help="number of samples for each iteration")
>>>>>>> dbd4d62f0278ccf3c5c041c285abf3c064bf5f9a
    parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate")
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)

    # DONOTCHANGE: They are reserved for nsml
    parser.add_argument("--pause", type=int, default=0)
    parser.add_argument('--mode', type=str, default='train', help='submit일때 test로 설정됩니다.')
    parser.add_argument('--iteration', type=str, default='0', help='fork 명령어를 입력할때의 체크포인트로 설정됩니다.')    
    args = parser.parse_args()
    return args

<<<<<<< HEAD
def get_train_transform():
    return A.Compose([
        A.Resize(1024,1024),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.VerticalFlip(p=0.4),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def sub_main(opt):
    n_gpus = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, nprocs=n_gpus, args=(opt, n_gpus, ))

def main(gpu, opt, n_gpus):
    opt.dist_url = "tcp://127.0.0.1:3333"
    torch.cuda.empty_cache()
    torch.distributed.init_process_group(backend='nccl', init_method=opt.dist_url, world_size=n_gpus, rank=gpu)

    torch.manual_seed(41)
    num_class = 30 # 순수한 데이터셋 클래스 개수

    # define model
    model = build_model(num_classes=num_class+1) # 배경 class 포함 모델
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    if opt.pause:
        bind_model(model)
        nsml.paused(scope=locals())
    else:
=======
def main(opt):
    
    torch.manual_seed(123)
    num_class = 30 # 순수한 데이터셋 클래스 개수

    # baseline model
    dboxes = generate_dboxes()
    model = BASELINE_MODEL(num_classes=num_class+1) # 배경 class 포함 모델
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.937, 0.999))
    scheduler = None

    bind_model(model)

    if opt.pause:
        nsml.paused(scope=locals())
    else:
        # loss
        criterion = Loss(dboxes)

>>>>>>> dbd4d62f0278ccf3c5c041c285abf3c064bf5f9a
        # train data
        with open(os.path.join(DATASET_PATH, 'train', 'train_label'), 'r', encoding="utf-8") as f:
            train_data_dict = json.load(f)
            train_img_label = prepocessing(root_dir=os.path.join(DATASET_PATH, 'train', 'train_data'),\
<<<<<<< HEAD
                label_data=train_data_dict, input_size=(IMAGE_SIZE, IMAGE_SIZE))
        

        train_data = Small_dataset(train_img_label, get_train_transform())
        sampler = DistributedSampler(train_data)

        train_params = {"batch_size": opt.batch_size,
                        "sampler": sampler,
=======
                label_data=train_data_dict, input_size=(300,300))
        
        train_params = {"batch_size": opt.batch_size,
                        "shuffle": True,
>>>>>>> dbd4d62f0278ccf3c5c041c285abf3c064bf5f9a
                        "drop_last": False,
                        "num_workers": opt.num_workers,
                        "collate_fn": collate_fn}

<<<<<<< HEAD
        train_loader = DataLoader(train_data, **train_params)

        model.cuda(gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
        bind_model(model)

        for epoch in range(0, opt.epochs):
            train_loss = train(model, train_loader, epoch, optimizer, scheduler, gpu)
=======
        # data loader
        train_data = Small_dataset(train_img_label, num_class, BaseTransform(dboxes))
        train_loader = DataLoader(train_data, **train_params)

        model.cuda()
        criterion.cuda()

        for epoch in range(0, opt.epochs):
            train_loss = train(model, train_loader, epoch, criterion, optimizer, scheduler)
>>>>>>> dbd4d62f0278ccf3c5c041c285abf3c064bf5f9a
            nsml.report(
                epoch=epoch,
                epoch_total=opt.epochs,
                batch_size=opt.batch_size,
                train_loss=train_loss)
            nsml.save(epoch)

if __name__ == "__main__":
    opt = get_args()
<<<<<<< HEAD
    sub_main(opt)
=======
    main(opt)
>>>>>>> dbd4d62f0278ccf3c5c041c285abf3c064bf5f9a

import os
import json
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
import glob

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
    parser.add_argument("--num-workers", type=int, default=4)

    # DONOTCHANGE: They are reserved for nsml
    parser.add_argument("--pause", type=int, default=0)
    parser.add_argument('--mode', type=str, default='train', help='submit일때 test로 설정됩니다.')
    parser.add_argument('--iteration', type=str, default='0', help='fork 명령어를 입력할때의 체크포인트로 설정됩니다.')    
    args = parser.parse_args()
    return args

def main(opt):

    classes = ['SD카드', '웹캠', 'OTP', '계산기', '목걸이', '넥타이핀', '십원', '오십원', '백원', '오백원', '미국지폐', '유로지폐', '태국지폐', '필리핀지폐',
            '밤', '브라질너트', '은행', '피칸', '호두', '호박씨', '해바라기씨', '줄자', '건전지', '망치', '못', '나사못', '볼트', '너트', '타카', '베어링']
            
    convert_to_coco_train(os.path.join(DATASET_PATH, 'train', 'train_label'),
            classes, coco_dict
    )
    convert_to_coco_valid(os.path.join(DATASET_PATH, 'train', 'train_label'),
            classes, coco_dict
    )

    CUR_PATH = os.getcwd()
    CFG_PATH = os.path.join("/app/configs/cascade_rcnn/cascade_rcnn_swin_tiny_fpn_1x_coco.py")
    PREFIX = os.path.join(DATASET_PATH, 'train', 'train_data')
    WORK_DIR = os.path.join('/app/work_dir')

    # config file 들고오기
    cfg = Config.fromfile(CFG_PATH)

    cfg.data.train.classes = classes
    cfg.data.train.img_prefix = PREFIX
    cfg.data.train.ann_file = CUR_PATH + "/all_train.json"

    cfg.data.val.classes = classes
    cfg.data.val.img_prefix = PREFIX
    cfg.data.val.ann_file = CUR_PATH + "/valid.json"

    # data
    cfg.data.samples_per_gpu = opt.batch_size
    cfg.data.workers_per_gpu = 4

    cfg.seed = 42
    cfg.gpu_ids = [0]
    cfg.work_dir = WORK_DIR
    cfg.runner.max_epochs = 10
    cfg.rtotal_epochs = 10
    cfg.optimizer = dict(type='Adam', lr=opt.lr, weight_decay=0.0001)

    cfg.lr_config = dict(
        policy='CosineAnnealing', # The policy of scheduler, also support CosineAnnealing, Cyclic, etc. Refer to details of supported LrUpdater from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9.
        by_epoch=False,
        warmup='linear', # The warmup policy, also support `exp` and `constant`.
        warmup_iters=500, # The number of iterations for warmup
        warmup_ratio=0.001, # The ratio of the starting learning rate used for warmup
        min_lr=1e-04)

    cfg.log_config.interval = 600
    cfg.checkpoint_config.interval = 1
    cfg.log_config = {'hooks': [{'type': 'TextLoggerHook'}], 'interval': 600}
    # cfg.fp16 = dict(loss_scale=512.)
    cfg.model.pretrained = None

    model = build_detector(cfg.model)
    datasets = [build_dataset(cfg.data.train)]
    model.CLASSES = datasets[0].CLASSES

    bind_model(model)    

    train_detector(model, datasets[0], cfg, distributed=False, validate=True)

    nsml.save(0)

if __name__ == "__main__":
    opt = get_args()
    main(opt)
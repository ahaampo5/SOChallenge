import os
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader

import mmcv
import time

# baseline model
from src.utils import weights_to_cpu, get_state_dict
from src.dataset import collate_fn, Small_dataset, prepocessing,\
    coco_dict, convert_to_coco_train, convert_to_coco_valid,\
    convert_to_coco_test

# nsml
import nsml
from nsml import DATASET_PATH

import sys
from mmcv import Config
from mmcv.runner import load_checkpoint, get_dist_info, init_dist
from mmcv.parallel import MMDataParallel
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector, set_random_seed, init_detector, single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.utils import collect_env, get_root_logger


def bind_model(model):
    def save(dir_path, **kwargs):
        meta = {}
        meta.update(mmcv_version=mmcv.__version__, time=time.asctime())
        if hasattr(model, 'CLASSES') and model.CLASSES is not None:
            # save class name to the meta
            meta.update(CLASSES=model.CLASSES)
        checkpoint = {
            'meta': meta,
            'state_dict': weights_to_cpu(get_state_dict(model))
        }
        torch.save(checkpoint, os.path.join(dir_path, 'model.pt'))
        print("model saved!")

    def load(dir_path):
        checkpoint_path = os.path.join(dir_path, 'model.pt')
        checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
        print('model loaded!')

    def infer(test_img_path_list): # data_loader에서 인자 받음
        dir_name = os.path.dirname(test_img_path_list[0])
        classes = ['SD카드', '웹캠', 'OTP', '계산기', '목걸이', '넥타이핀', '십원', '오십원', '백원', '오백원', '미국지폐', '유로지폐', '태국지폐', '필리핀지폐',
            '밤', '브라질너트', '은행', '피칸', '호두', '호박씨', '해바라기씨', '줄자', '건전지', '망치', '못', '나사못', '볼트', '너트', '타카', '베어링']

        convert_to_coco_test(test_img_path_list, classes, coco_dict)

        CUR_PATH = os.getcwd()
        CFG_PATH = os.path.join("/app/configs/cascade_rcnn/cascade_rcnn_swin_tiny_fpn_1x_coco.py")
        PREFIX = dir_name
        WORK_DIR = os.path.join('/app/work_dir')

        cfg = Config.fromfile(CFG_PATH)

        cfg.data.test.classes = classes
        cfg.data.test.img_prefix = PREFIX
        cfg.data.test.ann_file = CUR_PATH + "/test.json"

        cfg.seed = 42
        cfg.gpu_ids = [0]

        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
                    dataset,
                    samples_per_gpu=1,
                    workers_per_gpu=cfg.data.workers_per_gpu,
                    dist=False,
                    shuffle=False
                    )

        model.CLASSES = dataset.CLASSES
        net = MMDataParallel(model.cuda(), device_ids=[0])

        class_num = 30

        result_dict = {}
        for test_img in test_img_path_list:
            file_name = test_img.split('/')[-1]
            result_dict[file_name] = []

        net.eval()

        for data, img in zip(data_loader, test_img_path_list):
            with torch.no_grad():
                result = net(return_loss=False, rescale=True, **data)[0]
            file_name = img.split('/')[-1]
            detections = []

            for j in range(class_num):
                try:
                    for o in result[j]:
                        detections.append([
                            j,
                            float(o[0]),
                            float(o[1]),
                            float(o[2]-o[0]),
                            float(o[3]-o[1]),
                            float(o[4])
                        ])
                except:
                    continue
            result_dict[file_name] = detections

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
    model = None
    if not opt.pause:
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

        distributed = False
        if distributed == True:
            init_dist('pytorch', **cfg.dist_params)
            # re-set gpu_ids with distributed training mode
            _, world_size = get_dist_info()
            cfg.gpu_ids = range(world_size)

        cfg.work_dir = WORK_DIR
        cfg.runner.max_epochs = 20
        cfg.rtotal_epochs = 20
        cfg.optimizer = dict(type='Adam', lr=opt.lr, weight_decay=0.0001)

        cfg.lr_config = dict(
            policy='CosineAnnealing', # The policy of scheduler, also support CosineAnnealing, Cyclic, etc. Refer to details of supported LrUpdater from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9.
            by_epoch=False,
            warmup='linear', # The warmup policy, also support `exp` and `constant`.
            warmup_iters=500, # The number of iterations for warmup
            warmup_ratio=0.001, # The ratio of the starting learning rate used for warmup
            min_lr=1e-07)

        cfg.log_config.interval = 600
        cfg.checkpoint_config.interval = 1
        cfg.log_config = {'hooks': [{'type': 'TextLoggerHook'}], 'interval': 600}
        cfg.model.pretrained = None

        model = build_detector(cfg.model)
        datasets = [build_dataset(cfg.data.train)]
        model.CLASSES = datasets[0].CLASSES

    else:
        CFG_PATH = os.path.join("/app/configs/cascade_rcnn/cascade_rcnn_swin_tiny_fpn_1x_coco.py")

        # config file 들고오기
        cfg = Config.fromfile(CFG_PATH)
        cfg.model.pretrained = None
        model = build_detector(cfg.model)
        
    bind_model(model)

    if opt.pause:
        nsml.paused(scope=locals())
    else: 
        train_detector(model, datasets[0], cfg, distributed=distributed, validate=True)
        nsml.save(10)
if __name__ == "__main__":
    opt = get_args()
    main(opt)
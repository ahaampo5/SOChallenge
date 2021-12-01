import numpy as np
import itertools
from math import sqrt
from tqdm import tqdm

import torch
import torch.nn.functional as F

# class ==========================
def train(model, train_loader, epoch, optimizer, scheduler, gpu):
    print(f'epoch : [{epoch}]')
    model.train()
    progress_bar = tqdm(train_loader)
    for i, (img, targets) in enumerate(progress_bar):
        if i==1:
            break
        img = list(image.float().cuda(gpu) for image in img)
        targets = [{k: v.cuda(gpu) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        loss_dict = model(img, targets)

        losses = sum(loss for loss in loss_dict.values())
        progress_bar.set_description("Epoch: {}. Loss: {:.5f}".format(epoch, losses.item()))

        losses.backward()
        optimizer.step()

    if scheduler is not None:
        scheduler.step()

    return float(losses.item()) # 에러 확인
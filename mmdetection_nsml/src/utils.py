import numpy as np
import itertools
from math import sqrt
from tqdm import tqdm

<<<<<<< HEAD
import torch
import torch.nn.functional as F

# class ==========================
def train(model, train_loader, epoch, optimizer, scheduler, gpu):
    print(f'epoch : [{epoch}]')
    model.train()
    progress_bar = tqdm(train_loader)
    for i, (img, targets) in enumerate(progress_bar):
        img = list(image.float().cuda(gpu) for image in img)
        targets = [{k: v.cuda(gpu) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        loss_dict = model(img, targets)

        losses = sum(loss for loss in loss_dict.values())
        progress_bar.set_description("Epoch: {}. Loss: {:.5f}".format(epoch, losses.item()))

        losses.backward()
=======
from mmcv.parallel import is_module_wrapper
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torchvision.ops.boxes import box_iou, box_convert
import torchvision.transforms as transforms


# class ==========================
class BaseTransform(object):
    def __init__(self, dboxes, size=(300, 300)):
        self.size = size
        self.dboxes = dboxes
        self.encoder = Encoder(self.dboxes)

        self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        self.img_trans = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            self.normalize
        ])

    def __call__(self, img, img_size, bboxes=None, labels=None, max_num=200):

        img = self.img_trans(img).contiguous()
        bboxes, labels = self.encoder.encode(bboxes, labels)

        return img, img_size, bboxes, labels

class Encoder(object):

    def __init__(self, dboxes):
        self.dboxes = dboxes(order="ltrb")
        self.dboxes_xywh = dboxes(order="xywh").unsqueeze(dim=0)
        self.nboxes = self.dboxes.size(0)
        self.scale_xy = dboxes.scale_xy
        self.scale_wh = dboxes.scale_wh

    def encode(self, bboxes_in, labels_in, criteria=0.5):

        ious = box_iou(bboxes_in, self.dboxes)
        best_dbox_ious, best_dbox_idx = ious.max(dim=0)
        best_bbox_ious, best_bbox_idx = ious.max(dim=1)

        # set best ious 2.0
        best_dbox_ious.index_fill_(0, best_bbox_idx, 2.0)

        idx = torch.arange(0, best_bbox_idx.size(0), dtype=torch.int64)
        best_dbox_idx[best_bbox_idx[idx]] = idx

        # filter IoU > 0.5
        masks = best_dbox_ious > criteria
        labels_out = torch.zeros(self.nboxes, dtype=torch.long)
        labels_out[masks] = labels_in[best_dbox_idx[masks]]
        bboxes_out = self.dboxes.clone()
        bboxes_out[masks, :] = bboxes_in[best_dbox_idx[masks], :]
        bboxes_out = box_convert(bboxes_out, in_fmt="xyxy", out_fmt="cxcywh")
        return bboxes_out, labels_out

    def scale_back_batch(self, bboxes_in, scores_in):
        """
            Do scale and transform from xywh to ltrb
            suppose input Nx4xnum_bbox Nxlabel_numxnum_bbox
        """
        if bboxes_in.device == torch.device("cpu"):
            self.dboxes = self.dboxes.cpu()
            self.dboxes_xywh = self.dboxes_xywh.cpu()
        else:
            self.dboxes = self.dboxes.cuda()
            self.dboxes_xywh = self.dboxes_xywh.cuda()

        bboxes_in = bboxes_in.permute(0, 2, 1)
        scores_in = scores_in.permute(0, 2, 1)

        bboxes_in[:, :, :2] = self.scale_xy * bboxes_in[:, :, :2]
        bboxes_in[:, :, 2:] = self.scale_wh * bboxes_in[:, :, 2:]

        bboxes_in[:, :, :2] = bboxes_in[:, :, :2] * self.dboxes_xywh[:, :, 2:] + self.dboxes_xywh[:, :, :2]
        bboxes_in[:, :, 2:] = bboxes_in[:, :, 2:].exp() * self.dboxes_xywh[:, :, 2:]
        bboxes_in = box_convert(bboxes_in, in_fmt="cxcywh", out_fmt="xyxy")

        return bboxes_in, F.softmax(scores_in, dim=-1)

    def decode_batch(self, bboxes_in, scores_in, nms_threshold=0.45, max_output=200):
        bboxes, probs = self.scale_back_batch(bboxes_in, scores_in)
        output = []
        for bbox, prob in zip(bboxes.split(1, 0), probs.split(1, 0)):
            bbox = bbox.squeeze(0)
            prob = prob.squeeze(0)
            output.append(self.decode_single(bbox, prob, nms_threshold, max_output))
        return output

    def decode_single(self, bboxes_in, scores_in, nms_threshold, max_output, max_num=200):
        bboxes_out = []
        scores_out = []
        labels_out = []

        for i, score in enumerate(scores_in.split(1, 1)):
            if i == 0:
                continue

            score = score.squeeze(1)
            mask = score > 0.05

            bboxes, score = bboxes_in[mask, :], score[mask]
            if score.size(0) == 0: continue

            score_sorted, score_idx_sorted = score.sort(dim=0)

            # select max_output indices
            score_idx_sorted = score_idx_sorted[-max_num:]
            candidates = []

            while score_idx_sorted.numel() > 0:
                idx = score_idx_sorted[-1].item()
                bboxes_sorted = bboxes[score_idx_sorted, :]
                bboxes_idx = bboxes[idx, :].unsqueeze(dim=0)
                iou_sorted = box_iou(bboxes_sorted, bboxes_idx).squeeze()
                # we only need iou < nms_threshold
                score_idx_sorted = score_idx_sorted[iou_sorted < nms_threshold]
                candidates.append(idx)

            bboxes_out.append(bboxes[candidates, :])
            scores_out.append(score[candidates])
            labels_out.extend([i] * len(candidates))

        if not bboxes_out:
            return [torch.tensor([]) for _ in range(3)]

        bboxes_out, labels_out, scores_out = torch.cat(bboxes_out, dim=0), \
                                             torch.tensor(labels_out, dtype=torch.long), \
                                             torch.cat(scores_out, dim=0)

        _, max_ids = scores_out.sort(dim=0)
        max_ids = max_ids[-max_output:]
        return bboxes_out[max_ids, :], labels_out[max_ids], scores_out[max_ids]

class DefaultBoxes(object):
    def __init__(self, fig_size, feat_size, steps, scales, aspect_ratios, scale_xy=0.1, scale_wh=0.2):

        self.feat_size = feat_size
        self.fig_size = fig_size

        self.scale_xy = scale_xy
        self.scale_wh = scale_wh

        self.steps = steps
        self.scales = scales

        fk = fig_size / np.array(steps)
        self.aspect_ratios = aspect_ratios

        self.default_boxes = []
        for idx, sfeat in enumerate(self.feat_size):

            sk1 = scales[idx] / fig_size  + 1e-9
            sk2 = scales[idx + 1] / fig_size  + 1e-9
            sk3 = sqrt(sk1 * sk2)
            all_sizes = [(sk1, sk1), (sk3, sk3)]

            for alpha in aspect_ratios[idx]:
                w, h = sk1 * (sqrt(alpha) + 1e-9), sk1 / (sqrt(alpha) + 1e-9)
                all_sizes.append((w, h))
                all_sizes.append((h, w))
            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat), repeat=2):
                    cx, cy = (j + 0.5) / fk[idx], (i + 0.5) / fk[idx]
                    self.default_boxes.append((cx, cy, w, h))

        self.dboxes = torch.tensor(self.default_boxes, dtype=torch.float)
        self.dboxes.clamp_(min=0, max=1)
        self.dboxes_ltrb = box_convert(self.dboxes, in_fmt="cxcywh", out_fmt="xyxy")

    def __call__(self, order="ltrb"):
        if order == "ltrb":
            return self.dboxes_ltrb
        else:  # order == "xywh"
            return self.dboxes

# function =======================
def generate_dboxes():
    figsize = 300
    feat_size = [38, 19, 10, 5, 3, 1]
    steps = [8, 16, 32, 64, 100, 300]
    scales = [21, 45, 99, 153, 207, 261, 315]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes

def train(model, train_loader, epoch, criterion, optimizer, scheduler):
    model.train()
    progress_bar = tqdm(train_loader)
    for i, (img, _, _, gloc, glabel) in enumerate(progress_bar):
        img = img.cuda()
        gloc = gloc.cuda()
        glabel = glabel.cuda()

        ploc, plabel = model(img)
        ploc, plabel = ploc.float(), plabel.float()
        gloc = gloc.transpose(1, 2).contiguous()

        optimizer.zero_grad()
        loss = criterion(ploc, plabel, gloc, glabel)
        progress_bar.set_description("Epoch: {}. Loss: {:.5f}".format(epoch, loss.item()))      
        loss.backward()
>>>>>>> dbd4d62f0278ccf3c5c041c285abf3c064bf5f9a
        optimizer.step()

    if scheduler is not None:
        scheduler.step()

<<<<<<< HEAD
    return float(losses.item()) # 에러 확인
=======
    return float(loss.item()) # 에러 확인


def _save_to_state_dict(module, destination, prefix, keep_vars):
    """Saves module state to `destination` dictionary.
    This method is modified from :meth:`torch.nn.Module._save_to_state_dict`.
    Args:
        module (nn.Module): The module to generate state_dict.
        destination (dict): A dict where state will be stored.
        prefix (str): The prefix for parameters and buffers used in this
            module.
    """
    for name, param in module._parameters.items():
        if param is not None:
            destination[prefix + name] = param if keep_vars else param.detach()
    for name, buf in module._buffers.items():
        # remove check of _non_persistent_buffers_set to allow nn.BatchNorm2d
        if buf is not None:
            destination[prefix + name] = buf if keep_vars else buf.detach()


def get_state_dict(module, destination=None, prefix='', keep_vars=False):
    """Returns a dictionary containing a whole state of the module.
    Both parameters and persistent buffers (e.g. running averages) are
    included. Keys are corresponding parameter and buffer names.
    This method is modified from :meth:`torch.nn.Module.state_dict` to
    recursively check parallel module in case that the model has a complicated
    structure, e.g., nn.Module(nn.Module(DDP)).
    Args:
        module (nn.Module): The module to generate state_dict.
        destination (OrderedDict): Returned dict for the state of the
            module.
        prefix (str): Prefix of the key.
        keep_vars (bool): Whether to keep the variable property of the
            parameters. Default: False.
    Returns:
        dict: A dictionary containing a whole state of the module.
    """
    # recursively check parallel module in case that the model has a
    # complicated structure, e.g., nn.Module(nn.Module(DDP))
    if is_module_wrapper(module):
        module = module.module

    # below is the same as torch.nn.Module.state_dict()
    if destination is None:
        destination = OrderedDict()
        destination._metadata = OrderedDict()
    destination._metadata[prefix[:-1]] = local_metadata = dict(
        version=module._version)
    _save_to_state_dict(module, destination, prefix, keep_vars)
    for name, child in module._modules.items():
        if child is not None:
            get_state_dict(
                child, destination, prefix + name + '.', keep_vars=keep_vars)
    for hook in module._state_dict_hooks.values():
        hook_result = hook(module, destination, prefix, local_metadata)
        if hook_result is not None:
            destination = hook_result
    return destination

def weights_to_cpu(state_dict):
    """Copy a model state_dict to cpu.
    Args:
        state_dict (OrderedDict): Model weights on GPU.
    Returns:
        OrderedDict: Model weights on GPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    # Keep metadata in state_dict
    state_dict_cpu._metadata = getattr(state_dict, '_metadata', OrderedDict())
    return state_dict_cpu
>>>>>>> dbd4d62f0278ccf3c5c041c285abf3c064bf5f9a

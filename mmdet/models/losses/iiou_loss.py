import math

import mmcv
import torch
import torch.nn as nn

from mmdet.core import bbox_overlaps
from ..builder import LOSSES
from .utils import weighted_loss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def iiou(pred, target):
    S_p = abs((pred[:,2] - pred[:,0]) * (pred[:,3] - pred[:,1]))
    S   = abs((target[:,2] - target[:,0]) * (target[:,3] - target[:,1]))

    overlap_x_max = torch.min(pred[:,2], target[:,2])
    overlap_x_min = torch.max(pred[:,0], target[:,0])
    overlap_y_max = torch.min(pred[:,3], target[:,3])
    overlap_y_min = torch.max(pred[:,1], target[:,1])

    overlap = abs((overlap_x_max - overlap_x_min).clamp(min = 0) * (overlap_y_max - overlap_y_min).clamp(min = 0))
    
    entry_x_max = torch.max(pred[:,2], target[:,2])
    entry_x_min = torch.min(pred[:,0], target[:,0])
    entry_y_max = torch.max(pred[:,3], target[:,3])
    entry_y_min = torch.min(pred[:,1], target[:,1])

    entry = abs((entry_y_max - entry_y_min) * (entry_x_max - entry_x_min))

    iou = overlap / (S_p + S - overlap)
    iiou = iou - (entry - (S_p + S - overlap)) / entry

    return iiou
    

@LOSSES.register_module()
class IIoULoss(nn.Module):
    """IIoULoss.

    Computing the IIoU loss between a set of predicted bboxes and target bboxes.

    """

    def __init__(self):
        super(IIoULoss, self).__init__()
        
    def forward(self, pred, target):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
    
        """
        loss = 1 - iiou(pred, target)
        return loss
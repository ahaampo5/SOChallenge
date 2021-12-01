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
    S_p = abs((pred[:2] - pred[:0]) * (pred[:3] - pred[:1]))
    S   = abs((target[:2] - target[:0]) * (target[:3] - target[:1]))

    overlap_x_max = min(pred[:2], target[:2])
    overlap_x_min = max(pred[:0], target[:0])
    overlap_y_max = min(pred[:3], target[:3])
    overlap_y_min = max(pred[:0], target[:0])

    overlap = abs((overlap_x_max - overlap_x_min).clamp(min = 0) * (overlap_y_max - overlap_y_min).clamp(min = 0))
    
    entry_x_max = max(pred[:2], target[:2])
    entry_x_min = min(pred[:0], target[:0])
    entry_y_max = max(pred[:3], target[:3])
    entry_y_min = min(pred[:1], target[:1])

    entry = abs((entry_y_max - entry_y_min) * (entry_x_max - entry_x_min))

    iou = overlap / (S_p + S - overlap)
    iiou = iou - (entry - (S_p + S - overlap)) / entry

    return iiou
    

@LOSSES.register_module()
class IIoULoss(nn.Module):
    """IoULoss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.

    Args:
        linear (bool): If True, use linear scale of loss instead of log scale.
            Default: False.
        eps (float): Eps to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
    """

    def __init__(self,
                 linear=False,
                 eps=1e-6,
                 reduction='mean',
                 loss_weight=1.0):
        super(IIoULoss, self).__init__()
        self.linear = linear
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # iou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = 1 - iiou(pred, target)
        return loss
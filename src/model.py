import torch
import torch.nn as nn

import torchvision
import torchvision.models as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes=31, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda(torch.cuda.current_device())
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class IIoULoss(nn.Module):
    def __init__(self):
        super(IIoULoss, self).__init__()

    def forward(self, pred, target):
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

        return 1 - iiou

def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.
    Arguments:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)
    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    labal_smooth_loss = CrossEntropyLabelSmooth(num_classes=31)
    classification_loss = labal_smooth_loss(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, -1, 4)

    box_loss = det_utils.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        size_average=False,
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss

def build_model(num_classes=31):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.fastrcnn_loss = fastrcnn_loss
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
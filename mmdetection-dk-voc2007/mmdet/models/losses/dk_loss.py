# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def dk_loss(pred,
            label,
            dk,
            weight=None,
            reduction='mean',
            avg_factor=None,
            class_weight=None,
            ignore_index=-100,
            avg_non_ignore=False):
    """Calculate the dk_loss

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored.
            If None, it will be set to default value. Default: -100.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.

    Returns:
        torch.Tensor: The calculated loss
    """
    # print('--------------------------dk--------------------------------',len(dk),dk[:10]) 
    num_classes = pred.size(1)
    target = F.one_hot(label, num_classes=num_classes)
    
    pred_softmax = F.softmax(
                pred, dim=-1) if pred is not None else None

    target = target.type_as(pred_softmax)
    # print('---------------------((1 - pred_softmax) * target)-------------',((1 - pred_softmax) * target))
    pt = ((1 - pred_softmax) * target).sum(dim=1)
    # print('--------------------pt------------',pt)

    CE_weight = dk.to(device)  * pt.to(device)
    
    # The default value of ignore_index is the same as F.cross_entropy
    ignore_index = -100 if ignore_index is None else ignore_index

    # element-wise losses
    loss = F.cross_entropy(
        pred,
        label,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)
    # print('---------loss---------',loss)
    
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss = CE_weight * loss
    
    # average loss over non-ignored elements
    # pytorch's official cross_entropy average loss over non-ignored elements
    # refer to https://github.com/pytorch/pytorch/blob/56b43f4fec1f76953f15a627694d4bba34588969/torch/nn/functional.py#L2660  # noqa
    if (avg_factor is None) and avg_non_ignore and reduction == 'mean':
        avg_factor = label.numel() - (label == ignore_index).sum().item()

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def _expand_onehot_labels(labels, label_weights, label_channels, ignore_index):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(
        valid_mask & (labels < label_channels), as_tuple=False)

    if inds.numel() > 0:
        bin_labels[inds, labels[inds]] = 1

    valid_mask = valid_mask.view(-1, 1).expand(labels.size(0),
                                               label_channels).float()
    if label_weights is None:
        bin_label_weights = valid_mask
    else:
        bin_label_weights = label_weights.view(-1, 1).repeat(1, label_channels)
        bin_label_weights *= valid_mask

    return bin_labels, bin_label_weights, valid_mask


def overlap_area(box: torch.Tensor, boxes: torch.Tensor):
    """ 计算多个边界框的每个边界框与一个特定边界框的重叠面积占自己的比例
    Parameters
    ----------
    box: Tensor of shape `(4, )`
        一个边界框
    boxes: Tensor of shape `(n, 4)`
        多个边界框
    Returns
    -------
    overlap_area: Tensor of shape `(n, )`
        重叠面积比例
    """
    # 计算交集
    xy_max = torch.min(boxes[:, 2:], box[2:])
    xy_min = torch.max(boxes[:, :2], box[:2])
    inter = torch.clamp(xy_max - xy_min, min=0)
    inter = inter[:, 0] * inter[:, 1]

    # 计算并集
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    return inter / area_boxes


def dk_bbox(det_bboxes, dk_bboxes, inds, labels, gt_label, num_classes):
    dk_num = torch.ones(len(gt_label), dtype=torch.int)  # 每个检测狂对应的dk值
    # before = float(datetime.now().strftime('%Y%m%d%H%M%S.%f'))
    for i in range(len(dk_bboxes)):
        overlap_area_ration = overlap_area(dk_bboxes[i, :4].to(device), det_bboxes.to(device))

        j = 0
        label = torch.zeros(num_classes, dtype=torch.int)  # 用于标记出现过的种类
        label[gt_label[i]] = 1
        if len(overlap_area_ration) > 0:
            max_cro=torch.max(overlap_area_ration)
            if max_cro > 0.95:
                # print(overlap_area_ration)
                while j < len(det_bboxes) and overlap_area_ration[j] > 0.95 and overlap_area_ration[j] < 1.00: # 可被认定为预测同一区域
                    if torch.count_nonzero(label) == 3:
                        break
                    # print('----------------inds[j]--------------------',labels[inds[j]][0])
                    if label[labels[inds[j]][0]] == 0:  # 此类别未被记录过
                        label[labels[inds[j]][0]] = 1
                    j = j + 1
                # print('----------------label--------------------',label)
                dk_num[i] = torch.count_nonzero(label)
    # print('----------------dk_num--------------------',dk_num)
    # after = float(datetime.now().strftime('%Y%m%d%H%M%S.%f'))
    # print('--------------time-------------', after - before)
    return dk_num


def cal_dk(bboxes, scores, gt_label):
    num_classes = scores.size(1)  # 加上背景类,总类别数
    
    if bboxes.shape[1] > 4:
        bboxes = bboxes.view(scores.size(0), -1, 4)
    else:
        bboxes = bboxes[:, None].expand(
            scores.size(0), num_classes - 1, 4)
    scores = scores[:, :-1]  # 去除背景类得分
    
    labels = torch.arange(num_classes - 1, dtype=torch.int, device=scores.device)
    labels = labels.view(1, -1).expand_as(scores)
    
    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)
    # print('------------------bboxes,scores,labels----------------------',len(bboxes),len(scores),len(labels))
    # print('------------------gt_label----------------------',len(gt_label),gt_label)
    
    # 比较的检测框(真实类别对应的检测框)
    if gt_label[0] != num_classes -1:
        mask = gt_label < (num_classes -1)
        # print('------------------mask ---------------------',mask )
        gt_index=torch.max(torch.nonzero(mask)) + 1
        dk_bboxes = torch.zeros(gt_index, 4, dtype=torch.float)  
        for i in range(gt_index):
            dk_bboxes[i] = bboxes[i * (num_classes -1) + gt_label[i]]
        # print('---------------dk_bboxes-------------', len(dk_bboxes))
        
        # print('------------------scores----------------------',len(scores),scores)
        # 被比较的检测框
        # print('--------------------torch.max(scores)-----------------',torch.max(scores))
        mask = scores> 0.05
        inds = torch.nonzero(mask)
        det_bboxes = torch.zeros(len(inds), 4, dtype=torch.float)  
        for i in range(len(inds)):
            det_bboxes[i] = bboxes[inds[i]]
        # print('---------------det_bboxes-------------', len(det_bboxes),scores[inds])
    
        # dk_num_before = int(time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))
        dk_num = dk_bbox(det_bboxes, dk_bboxes, inds, labels, gt_label, num_classes)
        # dk_num_after = int(time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))
        # print('-----------dk_num_time------------------', dk_num_after-dk_num_before)
    else:
        dk_num = torch.ones(len(gt_label), dtype=torch.int)
    return dk_num


@LOSSES.register_module()
class DKLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 ignore_index=None,
                 loss_weight=1.0,
                 avg_non_ignore=False):
        """dkLoss.

        Args:
            use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to False.
            use_mask (bool, optional): Whether to use mask cross entropy loss.
                Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            ignore_index (int | None): The label index to be ignored.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
            avg_non_ignore (bool): The flag decides to whether the loss is
                only averaged over non-ignored targets. Default: False.
        """
        super(DKLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.avg_non_ignore = avg_non_ignore
        if ((ignore_index is not None) and not self.avg_non_ignore
                and self.reduction == 'mean'):
            warnings.warn(
                'Default ``avg_non_ignore`` is False, if you would like to '
                'ignore the certain label and average loss over non-ignore '
                'labels, which is the same with PyTorch official '
                'cross_entropy, set ``avg_non_ignore=True``.')

        self.cls_criterion = dk_loss

    def extra_repr(self):
        """Extra repr."""
        s = f'avg_non_ignore={self.avg_non_ignore}'
        return s

    def forward(self,
                cls_score,
                label,
                proposal_index,
                aug_bboxes,
                aug_scores,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=None,
                **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The method used to reduce the
                loss. Options are "none", "mean" and "sum".
            ignore_index (int | None): The label index to be ignored.
                If not None, it will override the default value. Default: None.
        Returns:
            torch.Tensor: The calculated loss.
        """
        # print('---------------------dk_proposal_index------------',proposal_index)
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if ignore_index is None:
            ignore_index = self.ignore_index

        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(
                self.class_weight, device=cls_score.device)
        else:
            class_weight = None
            
        dk = torch.ones(proposal_index[-1], dtype=torch.int)
        # 对每一个batch内的每个图像单独计算dk,每张照片采样512个建议框
        for i in range(len(proposal_index)-1):#去除0下标的记录
            if torch.max(aug_scores[proposal_index[i]:proposal_index[i+1]]) > 0.90:
                gt_label = label[proposal_index[i]:proposal_index[i+1]]  # 真实标签
                # print('------------------gt_label-----------------',aug_bboxes[proposal_index[i]:proposal_index[i+1]], aug_scores[proposal_index[i]:proposal_index[i+1]])
                dk[proposal_index[i]:proposal_index[i+1]] = cal_dk(aug_bboxes[proposal_index[i]:proposal_index[i+1]], aug_scores[proposal_index[i]:proposal_index[i+1]], gt_label)
                
        # print('--------------------------dk--------------------------------',len(dk),dk[:10])    
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            dk,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=ignore_index,
            avg_non_ignore=self.avg_non_ignore,
            **kwargs)
        return loss_cls

# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np
from mmcv.ops.nms import batched_nms

from mmdet.core.bbox.iou_calculators import bbox_overlaps


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
    # print('交集', inter)

    # 计算并集
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    # area_box = (box[2]-box[0])*(box[3]-box[1])
    # print('面积', area_boxes)

    return inter / area_boxes

def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   #epoch=-1,
                   score_factors=None,
                   return_inds=False):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_cfg (dict): a dict that contains the arguments of nms operations
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
        score_factors (Tensor, optional): The factors multiplied to scores
            before applying NMS. Default to None.
        return_inds (bool, optional): Whether return the indices of kept
            bboxes. Default to False.

    Returns:
        tuple: (dets, labels, indices (optional)), tensors of shape (k, 5),
            (k), and (k). Dets are boxes with scores. Labels are 0-based.
    """
    # print('mu_nms-epoch', epoch)
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4)

    scores = multi_scores[:, :-1]

    labels = torch.arange(num_classes, dtype=torch.long, device=scores.device)
    labels = labels.view(1, -1).expand_as(scores)

    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        # remove low scoring boxes
        valid_mask = scores > score_thr
    # multiply score_factor after threshold to preserve more bboxes, improve
    # mAP by 1% for YOLOv3
    if score_factors is not None:
        # expand the shape to match original shape of score
        score_factors = score_factors.view(-1, 1).expand(
            multi_scores.size(0), num_classes)
        score_factors = score_factors.reshape(-1)
        scores = scores * score_factors

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
        bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
    else:
        # TensorRT NMS plugin has invalid output filled with -1
        # add dummy data to make detection output correct.
        bboxes = torch.cat([bboxes, bboxes.new_zeros(1, 4)], dim=0)
        scores = torch.cat([scores, scores.new_zeros(1)], dim=0)
        labels = torch.cat([labels, labels.new_zeros(1)], dim=0)

    if bboxes.numel() == 0:
        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS '
                               'as it has not been executed this time')
        dets = torch.cat([bboxes, scores[:, None]], -1)
        if return_inds:
            return dets, labels, inds
        else:
            return dets, labels

    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)
    
    # num = 0
    # dim0, dim1 = dets.shape
    # flitercross_index = [True for x in range(dim0)]
    # for i in range(dim0):
    #     overlap_area_ration = overlap_area(dets[i, :4], dets[:, :4])
    #     org_score = dets[i][4]
    #     if flitercross_index[i] and org_score >= 0.30:
    #         org_x1 = dets[i][0]
    #         org_y1 = dets[i][1]
    #         org_x2 = dets[i][2]
    #         org_y2 = dets[i][3]
    #         org_area = 1
    #         j = i + 1
    #         while j < dim0:  # 未被定义为存在包含关系或非包含关系的预选框
    #             if 0.95 < overlap_area_ration[j] < 1.00:
    #                 if labels[keep[i]] != labels[keep[j]] and org_score - dets[j][
    #                     4] >= 0.30:  # 标签不同的两个建议框,且两个建议框的得分差值大于10%
    #                     num = num + 1
    #                     labels[keep[j]] = labels[keep[i]]  # 修改标签
    #                     flitercross_index[j] = False
    #                     j = j + 1
    #                 else:
    #                     j = j + 1
    #             else:
    #                 j = j + 1
    #     else:
    #         continue
    # fliter_inds = torch.from_numpy(np.array(flitercross_index)).nonzero(as_tuple=False).squeeze(1)
    # dets, keep = dets[fliter_inds], keep[fliter_inds]

    # dim0, dim1 = dets.shape
    # flitercontain_index = [True for x in range(dim0)]
    # for i in range(dim0):
    #     org_score = dets[i][4]
    #     if flitercontain_index[i] and org_score >= 0.60:
    #         org_x1 = dets[i][0]
    #         org_y1 = dets[i][1]
    #         org_x2 = dets[i][2]
    #         org_y2 = dets[i][3]
    #         j = i + 1
    #         while j < dim0 and flitercontain_index[j]:  # 未被定义为存在包含关系或非包含关系的预选框
    #             if labels[keep[i]] != labels[keep[j]] and org_score - dets[j][4] >= 0.60:  # 标签不同的两个建议框,且两个建议框的得分差值大于20%
    #                 if (org_x1 <= dets[j][0] and org_y1 <= dets[j][1] and org_x2 >= dets[j][2] and org_y2 >=
    #                     dets[j][3]) or (
    #                         org_x1 >= dets[j][0] and org_y1 >= dets[j][1] and org_x2 <= dets[j][2] and org_y2 <=
    #                         dets[j][3]):  # 存在包含关系或被包含关系
    #                     labels[keep[j]] = labels[keep[i]]  # 修改标签
    #                     # dets[j][4] = org_score
    #                     flitercontain_index[j] = False
    #                     j = j + 1
    #                     num = num + 1
    #                 else:
    #                     j = j + 1
    #             else:
    #                 j = j + 1
    #     else:
    #         continue
    # print(num)
    # fliter_inds = torch.from_numpy(np.array(flitercontain_index)).nonzero(as_tuple=False).squeeze(1)
    # dets, keep = dets[fliter_inds], keep[fliter_inds]


    # y = copy.deepcopy(labels[keep])
    # # 如果存在多种标签,且同时包含置信度高的框和置信度低的检测框
    # if torch.min(y) != torch.max(y) and dets[0][4] > 0.80 and dets[-1][4] < 0.10:  # 如果存在多种标签,且同时包含置信度高的框和置信度低的检测框
    #     label_maxscore = y[0]  # 总体最高得分标签
    #     # print('-----label_maxscore--------', label_maxscore)
    #     label_count = torch.bincount(y)  # 每个标签出现的次数
    #     label_count_sum = torch.sum(label_count)
    #     print('---------------label_count,label_count_sum-----------------', label_count, label_count_sum)
    #     label_maxnum = label_count.argmax()  # 次数最多的标签的下标
    #     label_numsort = label_count.argsort()  # 标签按出现次数排序，输出下标
    #     # print('-----label_numsort--------', label_numsort)
    #     # print('-----label_minnum+label_maxnum--------', label_count[label_numsort[-1]] + label_count[label_numsort[-2]])
    #     if label_count[label_numsort[-1]] + label_count[label_numsort[-2]] == label_count_sum:
    #         print('只有两种标签')
    #     # print('-----label_maxnum--------', label_maxnum)
    #     label_maxnum_index = copy.deepcopy(label_count)  # 记录与次数最多的标签出现相同次数的标签
    #     label_maxnum_count = 0
    #     for i in range(len(label_count)):
    #         if label_count[i] == label_count[label_maxnum]:
    #             label_maxnum_index[label_maxnum_count] = i  # 具有相同次数的标签
    #             label_maxnum_count = label_maxnum_count + 1
    #     # print(label_maxnum_count)
    #     flag = 0
    #     if label_maxnum_count != 1:  # 出现次数最多的标签不是唯一的
    #         for label in y:
    #             for label_index in label_maxnum_index:
    #                 print(label_index)
    #                 if label_index == label:
    #                     label_maxnum = label_index
    #                     flag = 1
    #                     break
    #             # print('----------------flag--------------------',flag)
    #             if flag == 1:
    #                 break
    #     # print(
    #     #     '-----------label_count[label_maxnum], label_count_sum,label_count[label_maxnum].float('
    #     #     ')/label_count_sum.float()-----------------',
    #     #     label_count[label_maxnum], label_count_sum, label_count[label_maxnum].float() / label_count_sum.float())
    #     label_maxnum_ration = label_count[label_maxnum].float() / label_count_sum.float()
    #     if label_maxnum == label_maxscore and label_maxnum_ration >= 0.50:
    #         print('----------labels[keep]--------------------', y)
    #         print('1111111111111111111111111111111')
    #         # 划分数据集
    #         dataset = copy.deepcopy(dets)
    #         train_index = 0
    #         test_index = 0
    #         for i in range(len(y)):
    #             if dets[i][4] > 0.80:
    #                 train_index = train_index + 1
    #                 test_index = test_index + 1
    #             elif dets[i][4] > 0.10:
    #                 test_index = test_index + 1
    #             else:
    #                 break
    #         # print('------dataset------', dataset)
    #         # print(train_index, test_index)
    #         train_data = dataset[:train_index, :]
    #         test_data = dataset[test_index:, :]
    #         # print('-----train_data-----', train_data)
    #         # print('-----test_data-----', test_data)
    #         train_x = center_pointer(train_data)
    #         test_x = center_pointer(test_data)
    #         # test_x[:, 2] = test_data[:, 4]
    #         # print('-----train_x-----', train_x)  # 对应标签y[:train_index]
    #         # print('-----test_x-----', test_x)  # 对应标签y[test_index:]
    #         dist_matrix = euclidean_dist(test_x, train_x)
    #         # print('-------------------------dist_matrix-----------------------------------------', dist_matrix)
    #         for i in range(len(test_x)):
    #             test_y = knn_sort(dist_matrix[i], y[:len(train_x)], 5)
    #             if y[test_index + i] != test_y:
    #                 # print('----------y[test_index + i],test_y----------', y[test_index + i],
    #                 #       test_y)
    #                 y[test_index + i] = test_y
    #         labels[keep] = y
    #         print('----------labels[keep]_knn--------------------', labels[keep])
    
    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    if return_inds:
        return dets, labels[keep], inds[keep]
    else:
        return dets, labels[keep]


def fast_nms(multi_bboxes,
             multi_scores,
             multi_coeffs,
             score_thr,
             iou_thr,
             top_k,
             max_num=-1):
    """Fast NMS in `YOLACT <https://arxiv.org/abs/1904.02689>`_.

    Fast NMS allows already-removed detections to suppress other detections so
    that every instance can be decided to be kept or discarded in parallel,
    which is not possible in traditional NMS. This relaxation allows us to
    implement Fast NMS entirely in standard GPU-accelerated matrix operations.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class+1), where the last column
            contains scores of the background class, but this will be ignored.
        multi_coeffs (Tensor): shape (n, #class*coeffs_dim).
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        iou_thr (float): IoU threshold to be considered as conflicted.
        top_k (int): if there are more than top_k bboxes before NMS,
            only top top_k will be kept.
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept. If -1, keep all the bboxes.
            Default: -1.

    Returns:
        tuple: (dets, labels, coefficients), tensors of shape (k, 5), (k, 1),
            and (k, coeffs_dim). Dets are boxes with scores.
            Labels are 0-based.
    """

    scores = multi_scores[:, :-1].t()  # [#class, n]
    scores, idx = scores.sort(1, descending=True)

    idx = idx[:, :top_k].contiguous()
    scores = scores[:, :top_k]  # [#class, topk]
    num_classes, num_dets = idx.size()
    boxes = multi_bboxes[idx.view(-1), :].view(num_classes, num_dets, 4)
    coeffs = multi_coeffs[idx.view(-1), :].view(num_classes, num_dets, -1)

    iou = bbox_overlaps(boxes, boxes)  # [#class, topk, topk]
    iou.triu_(diagonal=1)
    iou_max, _ = iou.max(dim=1)

    # Now just filter out the ones higher than the threshold
    keep = iou_max <= iou_thr

    # Second thresholding introduces 0.2 mAP gain at negligible time cost
    keep *= scores > score_thr

    # Assign each kept detection to its corresponding class
    classes = torch.arange(
        num_classes, device=boxes.device)[:, None].expand_as(keep)
    classes = classes[keep]

    boxes = boxes[keep]
    coeffs = coeffs[keep]
    scores = scores[keep]

    # Only keep the top max_num highest scores across all classes
    scores, idx = scores.sort(0, descending=True)
    if max_num > 0:
        idx = idx[:max_num]
        scores = scores[:max_num]

    classes = classes[idx]
    boxes = boxes[idx]
    coeffs = coeffs[idx]

    cls_dets = torch.cat([boxes, scores[:, None]], dim=1)
    return cls_dets, classes, coeffs

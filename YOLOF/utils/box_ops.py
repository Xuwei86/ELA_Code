import torch
from torchvision.ops.boxes import box_area
import math


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

def CIoU_loss(pred, target):

    r"""`Implementation of paper `Enhancing Geometric Factors into
    Model Learning and Inference for Object Detection and Instance
    Segmentation <https://arxiv.org/abs/2005.03572>`_.

    Code is modified from https://github.com/Zzh-tju/CIoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    eps = 1e-9
    # overlap
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    # union
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]

    c2 = cw**2 + ch**2 + eps

    b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
    b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
    b2_x1, b2_y1 = target[:, 0], target[:, 1]
    b2_x2, b2_y2 = target[:, 2], target[:, 3]

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2))**2 / 4
    right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2))**2 / 4
    rho2 = left + right

    factor = 4 / math.pi**2
    v = factor * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

    with torch.no_grad():
        alpha = (ious > 0.5).float() * v / (1 - ious + v)

    # CIoU
    cious = ious - (rho2 / c2 + alpha * v)
    loss = 1 - cious.clamp(min=-1.0, max=1.0)
    return loss

def alpha_CIoU_loss(boxa, boxb):
    """
    # 当boxes与真实box重合时，且都在在中心点重合时，一个长宽比接近真实box，一个差异很大
    # 我们认为长宽比接近的是比较好的，损失应该是比较小的。所以ciou增加了对box长宽比的考虑
    ciou=iou+两个box中心点距离平方/最小闭包区域对角线距离平方+alpha*v
    loss=1-iou+两个box中心点距离平方/最小闭包区域对角线距离平方+alpha*v
    注意loss跟上边不一样，这里不是1-ciou
    v用来度量长宽比的相似性,4/(pi *pi)*(arctan(boxa_w/boxa_h)-arctan(boxb_w/boxb_h))^2
    alpha是权重值，衡量ciou公式中第二项和第三项的权重，
    alpha大优先考虑v，alpha小优先考虑第二项距离比,alpha = v / ((1 - iou) + v)。
    """
    alpha_ = 3
    eps = 1e-9
    # 求交集
    inter_x1, inter_y1 = torch.maximum(boxa[:, 0], boxb[:, 0]), torch.maximum(boxa[:, 1], boxb[:, 1])
    inter_x2, inter_y2 = torch.minimum(boxa[:, 2], boxb[:, 2]), torch.minimum(boxa[:, 3], boxb[:, 3])
    inter_h = torch.maximum(torch.tensor([0]).cuda(), inter_y2 - inter_y1)
    inter_w = torch.maximum(torch.tensor([0]).cuda(), inter_x2 - inter_x1)

    #inter_h = torch.max(inter_y2,inter_y1)
    #inter_w = torch.max(inter_x2, inter_x1)

    inter_area = inter_w * inter_h

    # 求并集
    union_area = ((boxa[:, 3] - boxa[:, 1]) * (boxa[:, 2] - boxa[:, 0])) + \
                 ((boxb[:, 3] - boxb[:, 1]) * (boxb[:, 2] - boxb[:, 0])) - inter_area + eps  # + 1e-8 防止除零

    # 求最小闭包区域的x1,y1,x2,y2
    ac_x1, ac_y1 = torch.minimum(boxa[:, 0], boxb[:, 0]), torch.minimum(boxa[:, 1], boxb[:, 1])
    ac_x2, ac_y2 = torch.maximum(boxa[:, 2], boxb[:, 2]), torch.maximum(boxa[:, 3], boxb[:, 3])
    # 把两个bbox的x1,y1,x2,y2转换成ctr_x,ctr_y
    boxa_ctrx, boxa_ctry = boxa[:, 0] + (boxa[:, 2] - boxa[:, 0]) / 2, boxa[:, 1] + (boxa[:, 3] - boxa[:, 1]) / 2
    boxb_ctrx, boxb_ctry = boxb[:, 0] + (boxb[:, 2] - boxb[:, 0]) / 2, boxb[:, 1] + (boxb[:, 3] - boxb[:, 1]) / 2
    boxa_w, boxa_h = boxa[:, 2] - boxa[:, 0], boxa[:, 3] - boxa[:, 1]
    boxb_w, boxb_h = boxb[:, 2] - boxb[:, 0], boxb[:, 3] - boxb[:, 1]

    # 求两个box中心点距离平方length_box_ctr，最小闭包区域对角线距离平方length_ac
    length_box_ctr = (boxb_ctrx - boxa_ctrx) * (boxb_ctrx - boxa_ctrx) + \
                     (boxb_ctry - boxa_ctry) * (boxb_ctry - boxa_ctry)
    length_box_ctr_2 = torch.pow(length_box_ctr,alpha_)
    length_ac = (ac_x2 - ac_x1) * (ac_x2 - ac_x1) + (ac_y2 - ac_y1) * (ac_y2 - ac_y1)
    length_ac_2 = torch.pow(length_ac+eps,alpha_)

    v = (4 / (math.pi * math.pi)) * (torch.atan(boxa_w / boxa_h) - torch.atan(boxb_w / boxb_h)) \
        * (torch.atan(boxa_w / boxa_h) - torch.atan(boxb_w / boxb_h))
    iou = inter_area / (union_area + 1e-8)
    iou2 = torch.pow(inter_area / (union_area + eps),alpha_)

    alpha = v / ((1 +eps)- iou + v)
    # ciou = iou - length_box_ctr / length_ac - alpha * v
    ciou_loss =iou2 - length_box_ctr_2 / length_ac_2 - torch.pow(alpha * v+eps,alpha_)
    return ciou_loss

if __name__ == '__main__':
    box1 = torch.tensor([[10, 10, 20, 20]])
    box2 = torch.tensor([[15, 15, 25, 25]])
    iou = box_iou(box1, box2)
    print(iou)
    giou = generalized_box_iou(box1, box2)
    print(giou)


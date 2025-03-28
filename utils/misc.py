import torch

def intersection_over_union(boxes_pred, boxes_gt):

    box1_x1 = boxes_pred[...,0:1]
    box1_x2 = boxes_pred[...,0:1] + boxes_pred[...,2:3]
    box1_y1 = boxes_pred[...,1:2]
    box1_y2 = boxes_pred[..., 1:2] + boxes_pred[...,3:4]

    box2_x1 = boxes_gt[...,0:1]
    box2_x2 = boxes_gt[...,0:1] + boxes_pred[...,2:3]
    box2_y1 = boxes_gt[...,1:2]
    box2_y2 = boxes_gt[..., 1:2] + boxes_pred[...,3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box1_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)
    



    
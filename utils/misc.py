import torch

def intersection_over_union(boxes_pred, boxes_gt):

    box1_x1 = boxes_pred[...,0:1]
    box1_x2 = boxes_pred[...,0:1] + boxes_pred[...,2:3]
    box1_y1 = boxes_pred[...,1:2]
    box1_y2 = boxes_pred[..., 1:2] + boxes_pred[...,3:4]

    box2_x1 = boxes_gt[...,0:1]
    box2_x2 = boxes_gt[...,0:1] + boxes_gt[...,2:3]
    box2_y1 = boxes_gt[...,1:2]
    box2_y2 = boxes_gt[..., 1:2] + boxes_gt[...,3:4]

    #print([box2_x1[0,3,4,0], box2_x2[0,3,4,0], box2_y1[0,3,4,0], box2_y2[0,3,4,0]])
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    #print([x1[0,3,4,0],x2[0,3,4,0],y1[0,3,4,0],y2[0,3,4,0]])
    
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    
    #print(f'Intersection: {intersection[0,3,4,0]}')
    
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    
    #print(f'box1_area: {box1_area[0,3,4,0]}')
    #print(f'box2_area: {box2_area[0,3,4,0]}')
    #print(f'Denom: {(box1_area[0,3,4,0] + box2_area[0,3,4,0] - intersection[0,3,4,0] + 1e-6)}')
    
    return intersection / (box1_area + box2_area - intersection + 1e-6)
    



    
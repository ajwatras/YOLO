import torch
import torchvision
import torch.nn as nn
from .misc import intersection_over_union


class YoloV1Loss(nn.Module):
    def __init__(self, grid_size, num_boxes, num_classes):
        super().__init__()
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes

        self.lambda_no_obj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # reshape network output
        predictions = predictions.reshape(-1, 
                                          self.grid_size, 
                                          self.grid_size, 
                                          self.num_boxes * 5 + self.num_classes)
        
        # Compute Responsible bounding boxes for each grid cell.
        iou_b1 = intersection_over_union(predictions[..., 0:4], target[..., 0:4])
        iou_b2 = intersection_over_union(predictions[..., 5:9], target[..., 0:4])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim = 0)
        iou_maxes, bestbox = torch.max(ious, dim = 0)

        # Identify Max IoU Bounding Boxes        
        bboxes = predictions[...,0:self.num_boxes*5]
        N,H,W,C = predictions.shape
        bboxes = bboxes.view(N,H,W,self.num_boxes, 5)
        bestbox = bestbox.expand(-1,-1,-1,5)
        
        selected_bboxes = torch.gather(bboxes, dim=3, index = bestbox.unsqueeze(3)).squeeze(3)
        
        # Identify Class Codes and ground truth object
        class_codes = predictions[...,(self.num_boxes * 5):]
        object_ind = target[..., 4]

        # Compute Class Loss
        class_loss = torch.sum(object_ind * torch.sum((class_codes - target[...,5:])**2, -1))    
        
        # Compute Detection Loss
        detection_weight = self.lambda_no_obj * (1 - object_ind) + object_ind
        det_loss = torch.sum(detection_weight * (object_ind - selected_bboxes[...,4])**2)

        # Compute Bbox Coord Loss
        selected_bboxes[...,2:4] = torch.sqrt(selected_bboxes[...,2:4])
        target_bbox = target[...,0:4].clone()
        target_bbox[...,2:4] = torch.sqrt(target_bbox[...,2:4])
        
        bbox_loss = torch.sum(object_ind * torch.sum((selected_bboxes[...,0:4] - target_bbox)**2, dim=-1))

        # Combine Losses
        return class_loss + det_loss + self.lambda_coord*bbox_loss
        
        loss = 0.0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Compute whether object is present
                if (target[i,j,4] == 0):
                    continue
                else:
                    # Compute responsible bbox
                    

                    bbox_loss = 0.0  # SSE(x,y,sqrt(w), sqrt(h))
                    bbox_loss = self.lambda_coord * bbox_loss
                
                    conf_loss = 0.0 # (conf_predict - conf_gt)
                    class_loss = 0.0 # (SSE(p_ci))

                    # Combine loss parameters
                    loss += bbox_loss + conf_loss + class_loss
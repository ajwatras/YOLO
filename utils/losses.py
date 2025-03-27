import torch
import torchvision
from misc import intersection_over_union

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
        loss = 0.0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Compute whether object is present
                if (target[i,j,5] == 0):
                    
                else:
                    # Compute responsible bbox
                    iou_b1 = intersection_over_union(# BBOXES GO HERE)
                    iou_b2 = intersection_over_union(# BBOXES GO HERE)
                    bbox_loss = 0.0  # SSE(x,y,sqrt(w), sqrt(h))
                    bbox_loss = self.lambda_coord * bbox_loss
                
                    conf_loss = 0.0 # (conf_predict - conf_gt)
                    class_loss = 0.0 # (SSE(p_ci))

                    # Combine loss parameters
                    loss += bbox_loss + conf_loss + class_loss
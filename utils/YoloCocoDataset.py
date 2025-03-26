import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection
from torchvision import tv_tensors

class YOLO_COCO_DATASET(Dataset):
    def __init__(self, img_dir, ann_file, transform = None, num_classes = 80, image_size = (448,448), grid_size = (64, 64)):
        self.coco_ds = CocoDetection(img_dir, ann_file)
        self.transform = transform
        self.num_classes = num_classes
        self.img_size = image_size
        self.grid_size = grid_size
        self.grid_dims = (image_size[1] // grid_size[1], image_size[0] // grid_size[0])

    def __len__(self):
        return len(self.coco_ds)
    
    def __getitem__(self, idx):
        img, anns = self.coco_ds[idx]
        bboxes = tv_tensors.BoundingBoxes([x['bbox'] for x in anns], format='XYWH', canvas_size = (img.size[1], img.size[0]))
        labels = [x['category_id'] for x in anns]
        label = torch.zeros(self.grid_dims[1], self.grid_dims[0], 2 * (5 + self.num_classes))
        # Apply Transformations
        if bboxes.numel() < 1:
            #print("No Bounding Box Found")
            img = self.transform(img)
            return img, label

        if (idx == 250):
            print(bboxes)

        if self.transform is not None:
            img, bboxes = self.transform(img, bboxes)
       # print(bboxes)

        _, img_height, img_width = img.shape
        # Define yolo annotations
        for bbox, class_code in zip(bboxes, labels):


            # Correct for out of image bounding boxes.
            if bbox[0] + bbox[2] > img_width:
                bbox[2] = img_width - bbox[0]
            if bbox[1] + bbox[3] > img_height:
                bbox[3] = img_height - bbox[1]

            # If Class Code is invalid, skip
            if class_code > self.num_classes:
                continue
            # If bounding box is invalid, skip.
            if (bbox[0] >= img_width) or (bbox[2] <= 0):
                continue
            if (bbox[1] >= img_height) or (bbox[3] <= 0):
                continue


            # Compute Grid cell containing the bbox center
            bbox_center = (bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2)

            grid_x = (bbox_center[0] // self.grid_size[0]).int()
            grid_y = (bbox_center[1] // self.grid_size[1]).int()
            
            # Compute new centered bounding box
            relative_bbox = [0,0,0,0] 
            relative_bbox[0] = (bbox[0] - grid_x * self.grid_size[0]) / self.grid_size[0]
            relative_bbox[1] = (bbox[1] - grid_y * self.grid_size[1]) / self.grid_size[1]
            relative_bbox[2] = bbox[2] / img_width
            relative_bbox[3] = bbox[3] / img_height


            # check if cell already has an object and offset idx if so. 
            if (label[grid_y, grid_x, 4] == 0):
                offset = 0
            else:
                offset = (5 + self.num_classes)

            # fill in appropriate cell:
            label[grid_y, grid_x, offset] = relative_bbox[0]
            label[grid_y, grid_x, offset + 1] = relative_bbox[1]
            label[grid_y, grid_x, offset + 2] = relative_bbox[2]
            label[grid_y, grid_x, offset + 3] = relative_bbox[3]
            label[grid_y, grid_x, offset + 4] = 1.0
            label[grid_y, grid_x, offset + 5 + class_code] = 1.0     

        return img, label
     


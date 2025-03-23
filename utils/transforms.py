import torch
import torchvision
import random
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.transforms import ColorJitter

class RandomCropWithBBox:
    def __init__(self, min_crop_ratio=0.5, max_crop_ratio = 1.0):
        self.min_crop_ratio = min_crop_ratio
        self.max_crop_ratio = max_crop_ratio

    def __call__(self, image: Image.Image, bbox):
        img_width, img_height = image.size
        
        # Define crop size
        crop_width = random.randint(int(img_width * self.min_crop_ratio), int(img_width * self.max_crop_ratio))
        crop_height = random.randint(int(img_height * self.min_crop_ratio), int(img_height * self.max_crop_ratio))
        print(f'Crop Width: {crop_width}, Crop Height: {crop_height}')

        # Randomly select crop origin
        crop_x_min = random.randint(0, img_width - crop_width)
        crop_y_min = random.randint(0, img_height - crop_height)
        print(f'Crop X Min: {crop_x_min}, Crop Y Min: {crop_y_min}')

        print(bbox)
        # Compute new bounding box
        bbox[0] = max(0, bbox[0] - crop_x_min)
        bbox[1] = max(0, bbox[1] - crop_y_min)
        
        if (bbox[0] + bbox[2]) > crop_width:
            bbox[2] = crop_width - bbox[0]
        if (bbox[1] + bbox[3]) > crop_height:
            bbox[3] = crop_height - bbox[1]
        print(f'bbox: {bbox}')

        cropped_image = F.crop(image, crop_y_min, crop_x_min, crop_height, crop_width)
        print(f'{cropped_image}')

        return cropped_image, bbox
    
    def __repr__(self):
        return f"{self.__class__.__name__}(min_crop_ratio={self.min_crop_ratio}, max_crop_ratio={self.max_crop_ratio})"
    

class RandomColorJitterWithBBox:
    def __init__(self, brightness = 0.5, contrast = 0, saturation = 0.5, hue = 0.0):
        self.brightness = brightness
        self.saturation = saturation
        self.contrast = contrast
        self.hue = hue
        self.transform = ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)

    def __call__(self, image, bbox):
        return self.transform(image), bbox
    
    def __repr__(self):
        return f"{self.__class__.__name__}(brightness={self.brightness}, contrast={self.contrast}, hue={self.hue})"
        

            
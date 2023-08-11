# this file is used to create a custom dataset for the scalpel dataset
# the dataset is created by using the json files and the images
# the json files are used to get the bounding boxes and the images are used to get the images
# we need to load images and corresponding bbox and class labels and then visualize them
import numpy as np
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import functional as F
import json
import os
from PIL import Image
import cv2
import albumentations as A

transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
       #A.CenterCrop(height=280, width=280, p=1),
       #A.Resize(height=512, width=512),
    ],
    bbox_params=A.BboxParams(format='coco', min_visibility=0.5, label_fields=['category_ids']),
)

class CustomDatasetAlbu(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir, transform=transform):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.max_objects = 5
        self.num_classes = 5
        self.class_id_dict= {'LANE DISSECTING FORCEP 2 IN 3 TOOTHED 5.5': 0,
                              'NEEDLE HOLDER MAYO HEGAR 7': 1,                             
                              'LAHEY FORCEP': 2,
                              'MAYO SCISSOR STRAIGHT GOLD HANDLED 5.5': 3,
                               'FORCEP LITTLEWOOD': 4}

        self.img_names = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        label_path = os.path.join(self.label_dir, self.img_names[idx].replace('.png', '.json'))

        # Load image and label
        img = Image.open(img_path).convert("RGB")
        with open(label_path, 'r') as f:
            label_data = json.load(f)

        boxes = []
        classes = []
        # Parse JSON for bounding box and class label
        for instrument in label_data['instruments']:
            bb = instrument['bbox']
            cl = instrument['name'].strip('"')
            #x1, y1, x2, y2 = bbox
                
            boxes.append(bb)
            classes.append(self.class_id_dict[cl])

        if self.transform:
            image_np = np.array(img)

            # Applies transformations on the image
            #augmented = self.transform(image=image_np)
            augmented = transform(image=image_np, bboxes=boxes, category_ids=classes)

            # Converts the NumPy array back to a PIL Image
            img = Image.fromarray(augmented['image'])
            img= transforms.ToTensor()(img)


        #Pad the bounding boxes and classes to a fixed length (maximum objects across all images).
        #  This way, each item will always have the same shape, allowing for batching.             
        boxes_tensor = torch.zeros((self.max_objects, 4))
        classes_tensor = torch.zeros(self.max_objects)
        
        # One-hot encoding
       # classes_onehot = torch.zeros(len(classes), self.num_classes)
        #for i, cls in enumerate(classes):
        #    classes_onehot[i, cls] = 1

        for i, (box, cls) in enumerate(zip(boxes, classes)):
            boxes_tensor[i] = torch.tensor(box, dtype=torch.float32)
            classes_tensor[i] = torch.tensor(cls, dtype=torch.int64)

        bboxes = torch.tensor(boxes, dtype=torch.float32)
        classes = torch.tensor(classes, dtype=torch.long)
        return img, (bboxes, classes)

        #return img, (boxes_tensor, classes_tensor)
        #return img, (bbox, cls)
   
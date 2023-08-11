# this file is used to create a custom dataset for the scalpel dataset
# the dataset is created by using the json files and the images
# the json files are used to get the bounding boxes and the images are used to get the images
# we need to load images and corresponding bbox and class labels and then visualize them
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import json
import os
from PIL import Image

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
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
            img = self.transform(img)

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
    
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

img_dir = '/home/fmr/Downloads/scalpel/rescale/images'
label_dir = '/home/fmr/Downloads/scalpel/rescale/jsons'

dataset = CustomDataset(img_dir=img_dir, label_dir=label_dir, transform=transform)

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_data(dataset, index):
    img, (bbox, cls) = dataset[index]
    #img = img.permute(1, 2, 0).numpy()  # Convert CxHxW format to HxWxC for visualization
    
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    
    for box, label in zip(bbox, cls):
        x, y, w, h = box
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x, y, str(label), color='white', bbox=dict(facecolor='red', alpha=0.5))
    
    plt.show()

# Visualizing the first 5 images and their bounding boxes
#for i in range(5):
    #visualize_data(dataset, i)

#train_size = int(0.8 * len(dataset))
#test_size = len(dataset) - train_size
#train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

#train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

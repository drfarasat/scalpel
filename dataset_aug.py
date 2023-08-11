# this file is used to create a custom dataset for the scalpel dataset
# the dataset is created by using the json files and the images
# the json files are used to get the bounding boxes and the images are used to get the images
# we need to load images and corresponding bbox and class labels and then visualize them
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import functional as F
import json
import os
from PIL import Image

class RandomCropWithBBox(transforms.RandomCrop):
    def __call__(self, sample):
        img, bbox = sample['image'], sample['bbox']

        # Apply Random Crop to the image and adjust bounding boxes
        i, j, h, w = self.get_params(img, self.size)
        img = F.crop(img, i, j, h, w)

        # Adjust bbox coordinates. If a bounding box is outside the cropped area, it's removed.
        new_bboxes = []
        for box in bbox:
            x, y, width, height = box

            # Convert (x,y,w,h) to (xmin, ymin, xmax, ymax)
            x_min, y_min, x_max, y_max = x, y, x + width, y + height

            # Clip bbox coordinates to lie inside the cropped region
            x_min = min(max(x_min - j, 0), w)
            y_min = min(max(y_min - i, 0), h)
            x_max = max(min(x_max - j, w), 0)
            y_max = max(min(y_max - i, h), 0)

            # If the bounding box still has non-zero area, keep it
            if (x_max > x_min) and (y_max > y_min):
                new_bboxes.append([x_min, y_min, x_max - x_min, y_max - y_min])

        sample['bbox'] = new_bboxes
        sample['image'] = img

        return sample

class ResizeWithBBox:
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image, bbox = sample['image'], sample['bbox']
        w, h = image.size
        scale_w, scale_h = self.size[0] // w, self.size[1] // h

        new_bbox = [bbox[0] * scale_w, bbox[1] * scale_h, bbox[2] * scale_w, bbox[3] * scale_h]

        image = transforms.Resize(self.size)(image)

        return {'image': image, 'bbox': new_bbox}

#transform_set = transforms.Compose([
#    RandomCropWithBBox((500, 375), pad_if_needed=True, padding_mode='border'),
#    transforms.Lambda(lambda sample: {'image': transforms.RandomHorizontalFlip()(sample['image']), 'bbox': sample['bbox']}),
#    transforms.Lambda(lambda sample: {'image': transforms.RandomRotation(10)(sample['image']), 'bbox': sample['bbox']}),
#    transforms.Lambda(lambda sample: {'image': transforms.ToTensor()(sample['image']), 'bbox': sample['bbox']}),
#])


transform_set = transforms.Compose([
    RandomCropWithBBox((500, 375), pad_if_needed=True, padding_mode='border'),
    #transforms.Lambda(lambda sample: {'image': transforms.RandomHorizontalFlip()(sample['image']), 'bbox': [w - sample['bbox'][2], sample['bbox'][1], w - sample['bbox'][0], sample['bbox'][3]] if random.random() > 0.5 else sample['bbox']}),  # Assuming bbox is in [x1, y1, x2, y2] format
    # Skip random rotation for simplicity for now, but a similar approach can be used
    ResizeWithBBox((512, 512)),
    transforms.Lambda(lambda sample: {'image': transforms.ToTensor()(sample['image']), 'bbox': sample['bbox']}),
])

class CustomDatasetAugmented(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir, transform=transform_set):
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
            boxes.append(bb)
            classes.append(self.class_id_dict[cl])

        sample = {'image': img, 'bbox': boxes}
        
        if self.transform:
            sample = self.transform(sample)  # Note the change here

        boxes_tensor = torch.zeros((self.max_objects, 4), dtype=torch.float32)
        classes_tensor = torch.zeros(self.max_objects, dtype=torch.int64) - 1  # Using -1 for non-existent classes

        for i, (box, cls) in enumerate(zip(sample['bbox'], classes)):
            boxes_tensor[i] = torch.tensor(box, dtype=torch.float32)
            classes_tensor[i] = torch.tensor(cls, dtype=torch.int64)

        return sample['image'], (boxes_tensor, classes_tensor)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

img_dir = '/home/fmr/Downloads/scalpel/rescale/images'
label_dir = '/home/fmr/Downloads/scalpel/rescale/jsons'

dataset = CustomDatasetAugmented(img_dir=img_dir, label_dir=label_dir, transform=transform)

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

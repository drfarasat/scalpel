import torch
import os
import json
from PIL import Image
import numpy as np

class CustomDatasetTiles(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir, transform=None, tile_size=(128, 128)):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.max_objects = 5
        self.num_classes = 5
        self.tile_size = tile_size
        
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

        if self.transform:
            img = self.transform(img)

        all_tiles_features = []

        # Iterate over bounding boxes and extract tiles and their features
        for box in boxes:
            x1, y1, w, h = box
            x2 = x1 + w
            y2 = y1 + h 
            
            cropped_img = img.crop( [x1, y1, x2, y2])  # Assuming box is in the format [x1, y1, x2, y2]
            tiles = self.get_tiles(cropped_img)
            for tile in tiles:
                tile_feature = self.get_features(tile)
                all_tiles_features.append(tile_feature)
        
        all_tiles_features_tensor = torch.stack(all_tiles_features, dim=0)

        boxes_tensor = torch.zeros((self.max_objects, 4))
        classes_tensor = torch.zeros(self.max_objects)
        for i, (box, cls) in enumerate(zip(boxes, classes)):
            boxes_tensor[i] = torch.tensor(box, dtype=torch.float32)
            classes_tensor[i] = torch.tensor(cls, dtype=torch.int64)

        return img, (boxes_tensor, classes_tensor), all_tiles_features_tensor

    def get_tiles(self, cropped_img):
        w, h = cropped_img.size
        print('cropped image size', w, h)
        tw, th = self.tile_size
        tiles = []

        for i in range(0, w, tw):
            for j in range(0, h, th):
                tile = cropped_img.crop((i, j, i + tw, j + th))
                tiles.append(tile)
        
        return tiles

    def get_features(self, tile):
        # Simple feature extraction: mean pixel value.
        # This can be replaced with any other feature extraction method.
        return torch.tensor(np.array(tile).mean(axis=(0,1)), dtype=torch.float32)

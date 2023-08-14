import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms import ToPILImage

from dataset_tile import CustomDatasetTiles

# Instantiate the dataset
img_dir = '/home/fmr/Downloads/scalpel/rescale/images'
label_dir = '/home/fmr/Downloads/scalpel/rescale/jsons'

dataset = CustomDatasetTiles(img_dir=img_dir, label_dir=label_dir, tile_size=(64, 64))

# Load one sample from the dataset
img, (bboxes, classes), features = dataset[0]

# Convert tensor to PIL Image for visualization
if isinstance(img, torch.Tensor):
    img = ToPILImage()(img)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Display the original image with bounding boxes
ax[0].imshow(img)

for bbox in bboxes:
    if torch.sum(bbox) == 0:  # Ignore zero-padded boxes
        continue

    x1, y1, w, h = bbox
    x2 = x1 + w
    y2 = y1 + h 
    box_w = x2 - x1
    box_h = y2 - y1
    rect = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='r', facecolor='none')
    ax[0].add_patch(rect)

# Create an empty image canvas for tiles
#tile_canvas = torch.zeros_like(img)
tile_size = dataset.tile_size
tile_stride =(1,1)# dataset.tile_stride

# Create an empty numpy array for tiles
#tile_canvas = np.zeros((img.height, img.width, 3), dtype=np.uint8)

# Overlay the tiles onto the canvas
#for y in range(0, img.height - tile_size[1] + 1, tile_stride[1]):
#    for x in range(0, img.width - tile_size[0] + 1, tile_stride[0]):
#        tile = img.crop((x, y, x + tile_size[0], y + tile_size[1]))
#        #print(tile.size)
 #       tile_canvas[y:y+tile_size[1], x:x+tile_size[0], :] = np.array(tile)

tile_canvas = np.zeros((img.height, img.width, 3), dtype=np.uint8)

tile_size = tiles[0].size  # Assuming all tiles have the same size

y_offset = 0
for tile in tiles:
    if y_offset + tile_size[1] > img.height:
        y_offset = 0  # reset the y_offset if we've reached the image's height
    tile_canvas[y_offset:y_offset+tile_size[1], 0:tile_size[0], :] = np.array(tile)
    y_offset += tile_size[1]

ax[1].imshow(tile_canvas)

plt.tight_layout()
plt.show()

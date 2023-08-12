import torch
from torchvision.transforms import functional as F
from PIL import Image

# Assuming model is already loaded with trained weights
model = SimpleObjectDetectorRed()
model.load_state_dict(torch.load('path_to_saved_model.pth'))
model.eval()

# For CUDA acceleration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def get_bounding_boxes_and_classes(image_path, model, threshold=0.5):
    # Load image and preprocess
    image = Image.open(image_path).convert('RGB')
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        pred_bboxes, pred_classes = model(image_tensor)
    
    # Assuming pred_bboxes is of shape [batch, MAX_OBJECTS, 4]
    # and pred_classes is of shape [batch, MAX_OBJECTS, num_classes]
    pred_bboxes = pred_bboxes.squeeze(0)
    pred_classes_probs, pred_class_labels = torch.max(F.softmax(pred_classes.squeeze(0), dim=1), 1)
    
    # Filter out predictions with probability less than threshold
    valid_indices = pred_classes_probs > threshold
    valid_bboxes = pred_bboxes[valid_indices]
    valid_class_labels = pred_class_labels[valid_indices]
    
    return valid_bboxes, valid_class_labels

def extract_regions(image_path, bboxes):
    image = Image.open(image_path)
    regions = []
    for bbox in bboxes:
        # Assuming bbox is [xmin, ymin, xmax, ymax]
        region = image.crop(bbox.tolist())
        regions.append(region)
    return regions

image_path = 'path_to_your_image.jpg'
bboxes, class_labels = get_bounding_boxes_and_classes(image_path, model)

# Now, extract regions based on these bounding boxes
regions = extract_regions(image_path, bboxes)


# Now, you can process these regions further or save them as required
#for idx, region in enumerate(regions):
# #   region.save(f"region_{idx}.jpg")

def tile_image(image, tile_size, stride=None):
    if stride is None:
        stride = tile_size
        
    tiles = []
    for x in range(0, image.shape[1] - tile_size + 1, stride):
        tile = image[:, x:x+tile_size]
        tiles.append(tile)
        
    return tiles

def compute_features_for_tiles(tiles, feature_extractor):
    features = []
    for tile in tiles:
        tile = tile.unsqueeze(0)  # Add batch dimension
        feature = feature_extractor(tile)
        features.append(feature.squeeze().detach().cpu().numpy())
    return np.array(features)


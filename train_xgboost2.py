import torch
from torchvision.transforms import functional as F
from PIL import Image

from dataset import CustomDataset

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


import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
import xgboost as xgb
import numpy as np

# Initialize ResNet for feature extraction
model = resnet18(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])  # Remove the last FC layer
model.eval()

# CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# DataLoader and Dataset initialization
img_dir = "path_to_img_dir"
label_dir = "path_to_label_dir"
transform = None

dataset = CustomDataset(img_dir, label_dir, transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

def extract_and_aggregate(batch_imgs, batch_bboxes):
    all_aggregated_features = []
    
    for img, bboxes in zip(batch_imgs, batch_bboxes):
        aggregated_features_per_image = []

        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            cropped_img = F.crop(img, y1, x1, y2 - y1, x2 - x1)

            # Slice the cropped_img into tiles
            tile_size = 56  # Modify as needed
            tiles = [cropped_img[:, i:i+tile_size, j:j+tile_size] for i in range(0, cropped_img.shape[1], tile_size) for j in range(0, cropped_img.shape[2], tile_size)]
            
            # Extract features from each tile
            tile_features = []
            for tile in tiles:
                tile = tile.unsqueeze(0).to(device)
                features = model(tile)
                tile_features.append(features.cpu().squeeze().numpy())
            
            # Aggregating the features for a single cropped image
            aggregated_features = np.mean(tile_features, axis=0)
            aggregated_features_per_image.append(aggregated_features)

        all_aggregated_features.append(aggregated_features_per_image)
    
    return all_aggregated_features

#for batch_imgs, (batch_bboxes, batch_labels) in dataloader:
#    # Ensure that the input is a batch even if batch_size is 1
#    if len(batch_imgs.shape) == 3:
#        batch_imgs = batch_imgs.unsqueeze(0)
    
#    features = extract_and_aggregate(batch_imgs, batch_bboxes)

    # Here, you would then prepare the features for XGBoost and train or predict

# ... (rest of the code) ...

def train_xgboost(data, labels):
    # Here, 'data' is the aggregated features and 'labels' are the true labels
    # Prepare your DMatrix and train the model
    dtrain = xgb.DMatrix(data, label=labels)
    param = {'max_depth': 3, 'eta': 0.1, 'objective': 'multi:softmax', 'num_class': 5}
    num_round = 10
    bst = xgb.train(param, dtrain, num_round)
    return bst

def evaluate_xgboost(model, data, labels):
    # Here, 'data' is the aggregated features and 'labels' are the true labels
    # Predict using XGBoost
    dtest = xgb.DMatrix(data)
    preds = model.predict(dtest)
    # Calculate accuracy or any other metric
    correct = sum(1 for p, l in zip(preds, labels) if p == l)
    accuracy = correct / len(labels)
    return accuracy

# TRAINING
for batch_imgs, (batch_bboxes, batch_labels) in train_dataloader:
    # Ensure that the input is a batch even if batch_size is 1
    if len(batch_imgs.shape) == 3:
        batch_imgs = batch_imgs.unsqueeze(0)
    
    features = extract_and_aggregate(batch_imgs, batch_bboxes)

    # Here, you would train the XGBoost model with features
    model = train_xgboost(features, batch_labels)

# EVALUATION
accuracies = []
for batch_imgs, (batch_bboxes, batch_labels) in eval_dataloader:
    # Ensure that the input is a batch even if batch_size is 1
    if len(batch_imgs.shape) == 3:
        batch_imgs = batch_imgs.unsqueeze(0)
    
    features = extract_and_aggregate(batch_imgs, batch_bboxes)

    # Here, you would predict using the XGBoost model and compute accuracy
    acc = evaluate_xgboost(model, features, batch_labels)
    accuracies.append(acc)

# Compute the average accuracy over all batches
average_accuracy = sum(accuracies) / len(accuracies)
print(f"Average Accuracy: {average_accuracy}")

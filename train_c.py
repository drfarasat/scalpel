import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import CustomDataset
from dataset_aug import CustomDatasetAugmented
from loss import compute_loss
from network import SimpleObjectDetector, SimpleObjectDetectorDropout,SimpleObjectDetectorRed, SimpleObjectDetectorBN, \
SimpleObjectDetectorWithResnet, FPN


def draw_bboxes(image, bboxes, labels):
    """Draw bounding boxes on the image."""
    for bbox, label in zip(bboxes, labels):
        color = (0, 255, 0)  # Green
        x, y, w, h = bbox
        start_point = (int(x), int(y))
        end_point = (int(x + w), int(y + h))
        image = cv2.rectangle(image, start_point, end_point, color, 1)
        image = cv2.putText(image, str(label), start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

img_dir = '/home/fmr/Downloads/scalpel/rescale/images'
label_dir = '/home/fmr/Downloads/scalpel/rescale/jsons'

transform = transforms.Compose([
    #transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = CustomDataset(img_dir=img_dir, label_dir=label_dir, transform=transform)

#dataset = CustomDatasetAugmented(img_dir=img_dir, label_dir=label_dir)#, transform=transform)

# Splitting the dataset into training and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

BATCH_SIZE = 64
# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16)

# Model
#model = SimpleObjectDetector(input_width=224, input_height=224, num_classes=5, max_objects=5)
#model = SimpleObjectDetector()
#model = SimpleObjectDetectorWithResnet()
#model = FPN()
model = SimpleObjectDetectorRed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# Parameters

lr = 0.001

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)


# Training loop
NUM_EPOCHS = 1800
SAVE_EVERY = 50
MAX_OBJECTS =5
num_classes = 5

# Initializing lists to store losses
train_losses = []
valid_losses = []
for epoch in range(1, NUM_EPOCHS + 1):
# Training Loop

    model.train()
    running_loss = 0.0
    print('Epoch:', epoch + 1, 'of', NUM_EPOCHS, 'epochs')
    
    # Example usage during training:
    for images, (true_bboxes, true_classes) in train_loader:
        images, true_bboxes, true_classes = images.to(device), true_bboxes.to(device), true_classes.to(device)
        # ... continue with backpropagation and optimization ...


        # Forward + Backward + Optimize
        optimizer.zero_grad()
        pred_bboxes, pred_classes = model(images)
        loss = compute_loss(pred_bboxes, pred_classes, true_bboxes, true_classes)

        #outputs = model(images)
        #loss = criterion(outputs, bbox)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] - Loss: {epoch_loss:.4f}")

    # Save weights every 10 epochs
    if epoch % SAVE_EVERY == 0:
        torch.save(model.state_dict(), f"model_weights_epoch_{epoch}.pt")
        
        # Save few sample images with bounding boxes
        with torch.no_grad():
            sample_images = images[:5].cpu().numpy()  # taking first 5 images from the last batch
            pred_bboxes = pred_bboxes.view(-1, MAX_OBJECTS, 4)[:5].cpu().numpy()
            pred_classes = torch.argmax(pred_classes.view(-1, MAX_OBJECTS, num_classes)[:5], dim=-1).cpu().numpy()

            for idx, (image, bboxes, labels) in enumerate(zip(sample_images, pred_bboxes, pred_classes)):
                image = (image.transpose(1, 2, 0) * 255).astype(np.uint8).copy()
                image_with_bboxes = draw_bboxes(image, bboxes, labels)
                out_fname = f"pred_output/outputsample_epoch_{epoch}_img_{idx}.png"
                cv2.imwrite(out_fname, image_with_bboxes)
        # Simple Validation
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            for images, (true_bboxes, true_classes) in val_loader:
                images, true_bboxes, true_classes = images.to(device), true_bboxes.to(device), true_classes.to(device)
        
                pred_bboxes, pred_classes = model(images)
                loss = compute_loss(pred_bboxes, pred_classes, true_bboxes, true_classes)

                val_loss += loss.item() * images.size(0)
            
            val_epoch_loss = val_loss / len(val_loader.dataset)
            print(f"Validation Loss: {val_epoch_loss:.4f}")
            # Append the average loss at the end of each epoch
            train_losses.append(epoch_loss)
            valid_losses.append(val_epoch_loss)


# Visualization using matplotlib
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Training Loss", color="blue")
plt.plot(valid_losses, label="Validation Loss", color="red")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Using Python's built-in `pickle` module
import pickle

with open("train_losses.pkl", "wb") as f:
    pickle.dump(train_losses, f)

with open("valid_losses.pkl", "wb") as f:
    pickle.dump(valid_losses, f)
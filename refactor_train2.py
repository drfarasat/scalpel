import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from dataset import CustomDataset
from dataset_aug import CustomDatasetAugmented
from dataset_aug_albu import CustomDatasetAlbu
from loss import compute_loss, compute_loss_smooth_focal
from network import *
from network_attention import AttentionObjectDetector
import albumentations as A
import argparse
import pickle
import matplotlib.pyplot as plt

from network_fpn import FPNCATSimple


def get_args():
    parser = argparse.ArgumentParser(description="Train a Simple Object Detector")
    parser.add_argument("--img_dir", default='/home/fmr/Downloads/scalpel/rescale/images', type=str, help="Directory containing images")
    parser.add_argument("--label_dir", default='/home/fmr/Downloads/scalpel/rescale/jsons', type=str, help="Directory containing labels")
    parser.add_argument("--input_size", default=(512, 512), type=tuple, help="Input size for the model")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training")
    parser.add_argument("--num_workers", default=16, type=int, help="Number of workers for dataloaders")
    parser.add_argument("--num_epochs", default=8000, type=int, help="Number of epochs for training")
    parser.add_argument("--save_every", default=40, type=int, help="Save model every X epochs")
    parser.add_argument("--max_objects", default=5, type=int, help="Max number of objects for detection")
    parser.add_argument("--num_classes", default=5, type=int, help="Number of classes")
    parser.add_argument("--weights_path", default='', type=str, help="Path to weights for initializing model")
    return parser.parse_args()


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

def write_images(images, pred_bboxes, pred_classes, epoch, data_type="train"):
    sample_images = images[:5].cpu().numpy()
    pred_bboxes = pred_bboxes.view(-1, args.max_objects, 4)[:5].cpu().numpy()
    pred_classes = torch.argmax(pred_classes.view(-1, args.max_objects, args.num_classes)[:5], dim=-1).cpu().numpy()

    for idx, (image, bboxes, labels) in enumerate(zip(sample_images, pred_bboxes, pred_classes)):
        image = (image.transpose(1, 2, 0) * 255).astype(np.uint8).copy()
        image_with_bboxes = draw_bboxes(image, bboxes, labels)
        out_fname = f"pred_output/{data_type}/outputsample_epoch_{epoch}_img_{idx}.png"
        cv2.imwrite(out_fname, image_with_bboxes)

mean = [0.3250, 0.4593, 0.4189]
std = [0.1759, 0.1573, 0.1695]


def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, (true_bboxes, true_classes) in train_loader:
        images, true_bboxes, true_classes = images.to(device), true_bboxes.to(device), true_classes.to(device)
        optimizer.zero_grad()
        pred_bboxes, pred_classes = model(images)
        loss = compute_loss_smooth_focal(pred_bboxes, pred_classes, true_bboxes, true_classes, wbox=1.0)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(train_loader.dataset)

def validate(model, val_loader, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, (true_bboxes, true_classes) in val_loader:
            images, true_bboxes, true_classes = images.to(device), true_bboxes.to(device), true_classes.to(device)
            pred_bboxes, pred_classes = model(images)
            loss = compute_loss_smooth_focal(pred_bboxes, pred_classes, true_bboxes, true_classes, wbox=1)
            val_loss += loss.item() * images.size(0)
    return val_loss / len(val_loader.dataset)

def main(args):
    # Prepare data and dataloaders
    transform = transforms.Compose([transforms.ToTensor()])
    data_transforms = transforms.Compose([
        #transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    transform = data_transforms
    dataset = CustomDataset(img_dir=args.img_dir, label_dir=args.label_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Model setup
    model = FPNCATSimple()#input_size=args.input_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Parameters
    lr = 0.001
    #criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    # Weight initialization
    
    if os.path.exists(args.weights_path):
        model.load_state_dict(torch.load(args.weights_path))
        print(f"Loaded model weights from {args.weights_path}")

    # Training loop
    train_losses = []
    valid_losses = []
    """
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        running_loss = 0.0
        print('Epoch:', epoch + 1, 'of', args.num_epochs, 'epochs')
        
        for images, (true_bboxes, true_classes) in train_loader:
            images, true_bboxes, true_classes = images.to(device), true_bboxes.to(device), true_classes.to(device)
            optimizer.zero_grad()
            pred_bboxes, pred_classes = model(images)
            loss = compute_loss_smooth_focal(pred_bboxes, pred_classes, true_bboxes, true_classes, wbox=1.0)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{args.num_epochs}] - Loss: {epoch_loss:.4f}")

        # Save weights and outputs
        if epoch % args.save_every == 0:
            checkpoint_path = f"runs/ckpt_epoch_{epoch}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': loss,
            }, checkpoint_path)
            write_images(images, pred_bboxes, pred_classes, epoch, data_type="train")

        # Validation
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            for images, (true_bboxes, true_classes) in val_loader:
                images, true_bboxes, true_classes = images.to(device), true_bboxes.to(device), true_classes.to(device)
                pred_bboxes, pred_classes = model(images)
                loss = compute_loss_smooth_focal(pred_bboxes, pred_classes, true_bboxes, true_classes, wbox=1.0)
                val_loss += loss.item() * images.size(0)

            write_images(images, pred_bboxes, pred_classes, epoch, data_type="val")
                    
            val_epoch_loss = val_loss / len(val_loader.dataset)
            scheduler.step(val_epoch_loss)
            print(f"Validation Loss: {val_epoch_loss:.4f}")
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}, LR: {current_lr}")

            train_losses.append(epoch_loss)
            valid_losses.append(val_epoch_loss)
    """        
            

    # Visualization
    #plt.figure(figsize=(10, 5


    # Training loop
    train_losses = []
    valid_losses = []
    for epoch in range(1, args.num_epochs + 1):
        print('Epoch:', epoch, 'of', args.num_epochs, 'epochs')
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        print(f"Epoch [{epoch}/{args.num_epochs}] - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
        # Save weights and outputs
        if epoch % args.save_every == 0:
            checkpoint_path = f"runs/ckpt_epoch_{epoch}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': train_loss,
            }, checkpoint_path)
            with torch.no_grad():
                # Sample predictions
                def sampel_predictions(model, loader, device, epoch, data_type="train"):           
                    sample_images, (sample_bboxes, sample_classes) = next(iter(loader))
                    sample_images, sample_bboxes, sample_classes = sample_images.to(device), sample_bboxes.to(device), sample_classes.to(device)
                    pred_bboxes, pred_classes = model(sample_images[:5])
                    #pred_bboxes = pred_bboxes.view(-1, args.max_objects, 4).cpu().numpy()
                    #pred_classes = torch.argmax(pred_classes.view(-1, args.max_objects, args.num_classes), dim=-1).cpu().numpy()

                    write_images(sample_images, pred_bboxes, pred_classes, epoch, data_type="val")

                sampel_predictions(model, train_loader, device, epoch, data_type="train")
                sampel_predictions(model, val_loader, device, epoch, data_type="val")
     

        train_losses.append(train_loss)
        valid_losses.append(val_loss)
        scheduler.step(val_loss)

    # 



if __name__ == "__main__":
    args = get_args()
    main(args)
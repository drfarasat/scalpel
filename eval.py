import cv2
import numpy as np
import torch
from torchvision.ops import box_iou
import torch
from dataset import CustomDataset
from torch.utils.data import DataLoader
from torchvision import transforms

from network import FPN, FPNCAT
from network_fpn import FPNCATSimple

MAX_OBJECTS =5
num_classes=5

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

def calculate_iou(pred_boxes, true_boxes):
    
    return box_iou(pred_boxes, true_boxes)

def xywh_to_x1y1x2y2(boxes):
    """Convert [x, y, w, h] box format to [x1, y1, x2, y2] format."""
    x1 = boxes[..., 0]
    y1 = boxes[..., 1]
    x2 = x1 + boxes[..., 2]
    y2 = y1 + boxes[..., 3]
    return torch.stack([x1, y1, x2, y2], dim=-1)

mean = [0.3250, 0.4593, 0.4189]
std = [0.1759, 0.1573, 0.1695]

def denormalize(images, mean, std):
    mean = np.array(mean).reshape(1,3,1,1)
    std = np.array(std).reshape(1,3,1,1)
    return images * std + mean


def write_images(images, pred_bboxes, pred_classes, index, data_type="train"):
    sample_images = images.cpu().numpy()
    #need to denormalize images
    sample_images = denormalize(sample_images, mean, std)
    pred_bboxes = pred_bboxes.view(-1, MAX_OBJECTS, 4).cpu().numpy()
    pred_classes = torch.argmax(pred_classes.view(-1, MAX_OBJECTS, num_classes), dim=-1).cpu().numpy()

    for idx, (image, bboxes, labels) in enumerate(zip(sample_images, pred_bboxes, pred_classes)):

        image = (image.transpose(1, 2, 0) * 255).astype(np.uint8).copy()
        image_with_bboxes = draw_bboxes(image, bboxes, labels)
        out_fname = f"pred_output/{data_type}/validation_img_{index}_{idx}.png"
        cv2.imwrite(out_fname, image_with_bboxes)

def calculate_iou_fmr(pred_boxes, true_boxes):
    # Convert boxes from (x, y, w, h) to (x1, y1, x2, y2)
    pred_boxes = xywh_to_x1y1x2y2(pred_boxes)
    true_boxes = xywh_to_x1y1x2y2(true_boxes)
    
    inter_x1 = torch.max(pred_boxes[..., 0], true_boxes[..., 0])
    inter_y1 = torch.max(pred_boxes[..., 1], true_boxes[..., 1])
    inter_x2 = torch.min(pred_boxes[..., 2], true_boxes[..., 2])
    inter_y2 = torch.min(pred_boxes[..., 3], true_boxes[..., 3])
    
    inter_area = torch.max(inter_x2 - inter_x1 + 1, torch.tensor(0.)) * \
                 torch.max(inter_y2 - inter_y1 + 1, torch.tensor(0.))
    
    pred_area = (pred_boxes[..., 2] - pred_boxes[..., 0] + 1) * \
                (pred_boxes[..., 3] - pred_boxes[..., 1] + 1)
    
    true_area = (true_boxes[..., 2] - true_boxes[..., 0] + 1) * \
                (true_boxes[..., 3] - true_boxes[..., 1] + 1)
    
    union_area = pred_area + true_area - inter_area
    iou = inter_area / union_area
    
    return iou


def evaluate(model, dataloader, device):
    model.eval()

    total = 0
    total_correct_classifications = 0
    total_iou = 0
    iou_threshold = 0.5  # You can adjust this threshold as per your requirements

    with torch.no_grad():
        idx = 0
        for index ,(images, (true_bboxes, true_classes)) in enumerate(dataloader):
            
            images, true_bboxes, true_classes = images.to(device), true_bboxes.to(device), true_classes.to(device)

            pred_bboxes, pred_classes = model(images)
            write_images(images, pred_bboxes, pred_classes, index, data_type="val")
            # Convert predictions to proper shape
            pred_bboxes = pred_bboxes.view(-1, 5, 4)
            pred_classes = pred_classes.view(-1, 5, 5)

            # Calculate IoU
            iou = calculate_iou_fmr(pred_bboxes, true_bboxes)
            detected = (iou > iou_threshold).int()

            total_iou += iou.sum().item()
            
            # Checking class predictions
            _, predicted_class_ids = pred_classes.max(2)
            total_correct_classifications += (predicted_class_ids == true_classes).sum().item()

            total += true_classes.numel()

    accuracy = total_correct_classifications / total
    avg_iou = total_iou / total

    print(f"Class Accuracy: {accuracy:.4f}")
    print(f"Average IoU: {avg_iou:.4f}")

    model.train()

    return accuracy, avg_iou

def main():
    # Change these paths as per your setup
    img_dir = '/home/fmr/Downloads/scalpel/rescale/images'
    label_dir = '/home/fmr/Downloads/scalpel/rescale/jsons'
    mean = [0.3250, 0.4593, 0.4189]
    std = [0.1759, 0.1573, 0.1695]
    
    model_name = 'FPN'
    #uncomment below line for FPNCATSIMPLE
    #model_name = 'FPNCATSimple'


    if model_name == 'FPN':
        model = FPN()
        checkpoint_path = "savedweights/fpncatRes18/model_weights_epoch_4280.pt"
        transform = transforms.Compose([
        #transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ])
    
        dataset = CustomDataset(img_dir=img_dir, label_dir=label_dir, transform=transform)
   
    elif model_name == 'FPNCATSimple':
        model = FPNCATSimple()
        
        checkpoint_path = "savedweights/FPNCatSimple/model_weights_epoch_8000.pt"

        data_transforms = transforms.Compose([
        #transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])
        dataset = CustomDataset(img_dir=img_dir, label_dir=label_dir, transform=data_transforms)
    
      
    INPUT_SIZE = (512, 512 )
    BATCH_SIZE = 1
    NUM_WORKERS = 1

    # Splitting the dataset into training and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
    model.eval()
    evaluate(model, val_loader, device)

# here we define the entrypoint to the script
if __name__ == "__main__":
    main()

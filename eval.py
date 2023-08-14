import torch
from dataset import CustomDataset
from torch.utils.data import DataLoader
from torchvision import transforms

from network import FPN


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def evaluate_model(model, dataloader, iou_threshold=0.5):
    model.eval()

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for batch_imgs, (batch_bboxes, batch_classes) in dataloader:
        with torch.no_grad():
            predictions = model(batch_imgs)

        for idx, pred in enumerate(predictions):
            gt_boxes = batch_bboxes[idx]
            gt_classes = batch_classes[idx]

            for p_box, p_class, in zip(pred['boxes'], pred['labels']):#, pred['scores']):
                matched = False

                for gt_box, gt_class in zip(gt_boxes, gt_classes):
                    if compute_iou(p_box, gt_box) > iou_threshold and p_class == gt_class:
                        matched = True
                        break

                if matched:
                    true_positives += 1
                else:
                    false_positives += 1

            for gt_box, gt_class in zip(gt_boxes, gt_classes):
                matched = False

                for p_box, p_class, in zip(pred['boxes'], pred['labels'], pred['scores']):
                    if compute_iou(p_box, gt_box) > iou_threshold and p_class == gt_class:
                        matched = True
                        break

                if not matched:
                    false_negatives += 1

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score
img_dir = '/home/fmr/Downloads/scalpel/rescale/images'
label_dir = '/home/fmr/Downloads/scalpel/rescale/jsons'

transform = transforms.Compose([
    #transforms.Resize((224, 224)),
    transforms.ToTensor(),
    ])
dataset = CustomDataset(img_dir=img_dir, label_dir=label_dir, transform=transform)
    #dataset = CustomDatasetAlbu(img_dir=img_dir, label_dir=label_dir)#, transform=transform)

    
INPUT_SIZE = (512, 512 )
BATCH_SIZE = 16
NUM_WORKERS = 16

# Splitting the dataset into training and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

model = FPN()

# Load the trained model weights
checkpoint_path = "savedweights/fpncatRes18/model_weights_epoch_4280.pt"
model.load_state_dict(torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'))

# If you're loading the model for inference (not further training), set the model to eval mode
model.eval()
precision, recall, f1 = evaluate_model(model, val_loader)
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

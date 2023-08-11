import torch.nn.functional as F
import torch

def compute_loss(pred_bboxes, pred_classes, true_bboxes, true_classes):
    MAX_OBJECTS = 5
    num_classes = 5
    pred_bboxes = pred_bboxes.view(-1, MAX_OBJECTS, 4)
    pred_classes = pred_classes.view(-1, MAX_OBJECTS, num_classes)
    # Bounding box regression loss
    reg_loss = F.mse_loss(pred_bboxes, true_bboxes, reduction='mean')

    # Classification loss
    # We use BCEWithLogitsLoss because it combines a sigmoid layer and the BCE loss in one,
    # which makes it more numerically stable than using a plain sigmoid followed by BCE loss.
    cls_loss_fn = torch.nn.BCEWithLogitsLoss()
    cls_loss_fn = torch.nn.CrossEntropyLoss()
    cls_loss = cls_loss_fn(pred_classes, true_classes)

    # Combine the two losses
    # You can add weights here if you want to weigh the importance of one loss over the other
    #combined_loss = reg_loss + cls_loss

    #return combined_loss
    # Weights
    w_bbox = 1.0
    w_class =2
    
    return w_bbox * reg_loss + w_class * cls_loss
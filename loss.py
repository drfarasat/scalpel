import torch.nn.functional as F
import torch
from torch import nn

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
    #cls_loss_fn = torch.nn.BCEWithLogitsLoss()
    cls_loss_fn = torch.nn.CrossEntropyLoss()
    cls_loss = cls_loss_fn(pred_classes, true_classes)

    # Combine the two losses
    # You can add weights here if you want to weigh the importance of one loss over the other
    #combined_loss = reg_loss + cls_loss

    #return combined_loss
    # Weights
    w_bbox = 1.0
    w_class =1.8
    
    return w_bbox * reg_loss + w_class * cls_loss


class MultiClassFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(MultiClassFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Ensure targets are one-hot encoded
        if targets.ndimension() == 1:
            targets_one_hot = torch.zeros(targets.size(0), inputs.size(1), device=inputs.device)
            targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
            targets = targets_one_hot

        probs = F.softmax(inputs, dim=1)
        probs = (probs * targets).sum(1)
        probs = probs.clamp(min=1e-7)
        
        log_probs = -torch.log(probs)
        loss = self.alpha * (1 - probs) ** self.gamma * log_probs

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss

import torch.nn.functional as F

# Initialize the Focal Loss
focal_loss = MultiClassFocalLoss(gamma=2., alpha=0.25)

def compute_loss_smooth_focal(pred_bboxes, pred_classes, true_bboxes, true_classes):
    """
    predictions: Tuple containing bounding box and class predictions from the model.
    targets: Tuple containing true bounding boxes and class labels.
    """
    MAX_OBJECTS = 5
    num_classes = 5
    pred_bboxes = pred_bboxes.view(-1, MAX_OBJECTS, 4)
    pred_classes = pred_classes.view(-1, MAX_OBJECTS, num_classes)
   
    
    #pred_bboxes, pred_classes = predictions
    #true_bboxes, true_classes = targets

    # Regression Loss for Bounding Boxes
    bbox_loss = F.smooth_l1_loss(pred_bboxes, true_bboxes, reduction='mean')

    # Class Loss using Focal Loss
    class_loss = focal_loss(pred_classes, true_classes)

    # Combine the losses
    combined_loss = bbox_loss + 1.2 * class_loss

    return combined_loss

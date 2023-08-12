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


import torch.nn as nn
import torch.nn.functional as F

class MultiClassFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', class_weights=None):
        super(MultiClassFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights
        
        # If class weights are not provided, use uniform weights
        if self.class_weights is None:
            self.class_weights = torch.ones(5, dtype=torch.float32)

        # If using GPU
        if torch.cuda.is_available():
            self.class_weights = self.class_weights.cuda()

    def forward(self, inputs, targets):
        if targets.ndimension() == 1:
            targets_one_hot = torch.zeros(targets.size(0), inputs.size(1), device=inputs.device)
            targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
            targets = targets_one_hot

        probs = F.softmax(inputs, dim=1)
        probs = (probs * targets).sum(1)
        probs = probs.clamp(min=1e-7)
        
        # Incorporate class weights
        weights = self.class_weights[targets.argmax(1)]
        
        log_probs = -torch.log(probs)
        loss = self.alpha * (1 - probs) ** self.gamma * log_probs * weights

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
            
        return loss

# Initialize the Focal Loss with the class weights
weights = torch.tensor([1.0, 1.5, 1.5, 1.5, 1.0], dtype=torch.float32)# class 0 and 4 are easy to detect
focal_loss = MultiClassFocalLoss(gamma=2., alpha=0.25, class_weights=weights)

def compute_loss_smooth_focal(pred_bboxes, pred_classes, true_bboxes, true_classes, wbox=0.0):
    MAX_OBJECTS = 5
    num_classes = 5
    pred_bboxes = pred_bboxes.view(-1, MAX_OBJECTS, 4)
    pred_classes = pred_classes.view(-1, MAX_OBJECTS, num_classes)
    
    # Regression Loss for Bounding Boxes
    bbox_loss = F.smooth_l1_loss(pred_bboxes, true_bboxes, reduction='mean')

    # Adjust shapes for Focal Loss calculation
    pred_classes_flat = pred_classes.view(-1, num_classes)
    true_classes_flat = true_classes.view(-1)
    
    # Class Loss using Focal Loss
    class_loss = focal_loss(pred_classes_flat, true_classes_flat)
    
    combined_loss = wbox*bbox_loss + class_loss

    return combined_loss

import numpy as np
import torch
import torch.nn.functional as F
def dice_loss(pred, target, smooth=1):
    """
    Computes the Dice Loss for binary segmentation.
    Args:
        pred: Tensor of predictions (batch_size, 1, H, W).
        target: Tensor of ground truth (batch_size, 1, H, W).
        smooth: Smoothing factor to avoid division by zero.
    Returns:
        Scalar Dice Loss.
    """
    # Apply sigmoid to convert logits to probabilities
    if pred.max() > 1 or pred.min() < 0:
        pred = torch.sigmoid(pred)
    
    # Calculate intersection and union
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    
    # Compute Dice Coefficient
    dice = (2. * intersection + smooth) / (union + smooth)
    
    # Return Dice Loss
    return 1 - dice.mean()

def iou_score(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection) / (union)

def precision(y_true, y_pred):
    TP = np.logical_and(y_pred == 1, y_true == 1).sum()
    FP = np.logical_and(y_pred == 1, y_true == 0).sum()
    return TP / (TP + FP + 1e-6)

def recall(y_true, y_pred):
    TP = np.logical_and(y_pred == 1, y_true == 1).sum()
    FN = np.logical_and(y_pred == 0, y_true == 1).sum()
    return TP / (TP + FN + 1e-6)

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r + 1e-6)

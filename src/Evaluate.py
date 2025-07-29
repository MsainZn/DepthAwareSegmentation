import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
from ConcatModels import BaseUNet

def eval_model(model: BaseUNet, 
               dataloader: DataLoader,
               num_class: int,
               loss_fn: nn.Module,
               device: torch.device) -> dict:
    """Evaluate the model for one epoch and return loss, IoU, Dice, and Accuracy per class."""
    model.eval()
    model = model.to(device)

    running_loss = 0.0

    total_preds = []
    total_labels = []

    with torch.no_grad():
        for input_chnls, labels in dataloader:
            
            labels = labels.to(device)
            x = {k: v.to(device) for k, v in input_chnls.items()}

            # Forward pass
            outputs = model(x)

            # Compute the loss
            dpt = x['depth'] if 'depth' in x else None
            running_loss += loss_fn(outputs, labels, dpt).item()

            # running_loss += loss_fn(logits, labels).item()
            preds = model.predict_with_logits(outputs['seg'])
            
            # Store predictions and labels for metrics
            total_preds.append(preds)
            total_labels.append(labels)

    # Concatenate all batches
    total_preds = torch.cat(total_preds, dim=0)
    total_labels = torch.cat(total_labels, dim=0)

    # Compute segmentation metrics
    metrics = compute_segmentation_metrics(total_preds, total_labels, num_class, device)

    metrics["loss"] = running_loss / len(dataloader)

    return metrics

def compute_iou(preds: torch.Tensor, labels: torch.Tensor, num_class: int, device: torch.device) -> torch.Tensor:
    """Compute per-class Intersection over Union (IoU) on GPU."""
    intersection = torch.zeros(num_class, device=device)
    union = torch.zeros(num_class, device=device)

    for c in range(num_class):
        pred_c = (preds == c)
        label_c = (labels == c)

        intersection[c] += torch.sum(pred_c & label_c)
        union[c] += torch.sum(pred_c | label_c)

    iou = intersection / (union + 1e-7)
    return iou  # Keep as torch.tensor on GPU

def compute_dice(preds: torch.Tensor, labels: torch.Tensor, num_class: int, device: torch.device) -> torch.Tensor:
    """Compute per-class Dice coefficient on GPU."""
    tp = torch.zeros(num_class, device=device)     # True Positives
    fp_fn = torch.zeros(num_class, device=device)  # False Positives + False Negatives

    for c in range(num_class):
        pred_c = (preds == c)
        label_c = (labels == c)

        tp[c] += torch.sum(pred_c & label_c)
        fp_fn[c] += torch.sum(pred_c) + torch.sum(label_c)

    dice = (2 * tp) / (fp_fn + 1e-7)
    return dice  # Keep as torch.tensor on GPU

def compute_accuracy(preds: torch.Tensor, labels: torch.Tensor, num_class: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-class accuracy on GPU."""
    correct_pixels = torch.zeros(num_class, device=device)
    total_pixels = torch.zeros(num_class, device=device)

    for c in range(num_class):
        correct_pixels[c] += torch.sum(preds[labels == c] == c)
        total_pixels[c] += torch.sum(labels == c)

    accuracy = correct_pixels / (total_pixels + 1e-7)
    return accuracy, total_pixels  # Keep as torch.tensor on GPU

def compute_segmentation_metrics(
    preds: torch.Tensor, labels: torch.Tensor, num_class: int, device: torch.device, round_num:int=4
) -> Dict[str, float | List[float]]:
    """Compute IoU, Dice, and Accuracy metrics while keeping computations on GPU."""
    iou = compute_iou(preds, labels, num_class, device)
    dice = compute_dice(preds, labels, num_class, device)
    accuracy, total_pixels = compute_accuracy(preds, labels, num_class, device)

    # Compute class-weighted (balanced) metrics
    class_weights = total_pixels / total_pixels.sum()
    balanced_iou = torch.sum(iou * class_weights)
    balanced_dice = torch.sum(dice * class_weights)
    balanced_acc = torch.sum(accuracy * class_weights)

    # Convert to CPU and numpy at the last moment
    return {
    "balanced_iou":  round(balanced_iou.item(), round_num),
    "balanced_dice": round(balanced_dice.item(), round_num),
    "balanced_acc":  round(balanced_acc.item(), round_num),
    "iou_per_class": [round(x, round_num) for x in iou.cpu().numpy().tolist()],
    "dice_per_class": [round(x, round_num) for x in dice.cpu().numpy().tolist()],
    "acc_per_class": [round(x, round_num) for x in accuracy.cpu().numpy().tolist()]
}
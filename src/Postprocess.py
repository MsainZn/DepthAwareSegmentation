import torch
import os
from torch.optim import Optimizer
from torch.nn import Module
from typing import Dict, List, Tuple, Union
from torch.utils.data import Dataset
from Visualize import show_prd_vs_gt
from ConcatModels import BaseUNet
from torch.utils.data import Subset

def load_checkpoint(model: torch.nn.Module, 
                    model_path:str,
                    device: str,
                    ) -> torch.nn.Module:
    """Loads a model checkpoint from a file."""
    map_location=torch.device(device)
    checkpoint = torch.load(model_path, map_location=map_location)
    
    # Restore model and optimizer state dicts
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Checkpoint loaded from {model_path}")
    return model

def log_metrics(log_file: str, 
                iteration: int, 
                metric_train: Dict[str, Union[List[float], float]], 
                metric_test: Dict[str, Union[List[float], float]]) -> None:
    """Logs metrics for each iteration to a file."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "a") as f:
        f.write(f"Iteration {iteration}\n")
        f.write(f"Train Metrics: {metric_train}\n")
        f.write(f"Test Metrics: {metric_test}\n")
        f.write("----------------------------------\n")
        f.close()
    return

def evaluate_save_criterion(metric_train: Dict[str, List[float]], 
                            metric_test: Dict[str, List[float]], 
                            best_metrics: Dict[str, Union[List[float], float]]) -> bool:
    
    """Evaluates whether the model should be saved based on IoU improvements."""
    train_iou = metric_train['iou_per_class']
    test_iou = metric_test['iou_per_class']
    
    if len(train_iou) < 2 or len(test_iou) < 2:
        return False  # Not enough data to evaluate saving condition
    
    return (train_iou[-2] > best_metrics['train'][-2] and train_iou[-1] > best_metrics['train'][-1] and
            test_iou[-2] > best_metrics['test'][-2] and test_iou[-1] > best_metrics['test'][-1])

def save_checkpoint(model: Module,
                    model_name: str, 
                    save_dir: str, 
                    optimizer: Optimizer, 
                    iteration: int) -> bool:
    """Saves the model state if criteria are met."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(os.path.join(save_dir, f'{model_name}.pth'))
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, save_path)
    # print(f"Model saved at iteration {iteration}: {save_path}")
    return True

def show_prediction(dataset:Union[Subset, Dataset], model:BaseUNet):
    for i, (img, msk) in enumerate(dataset.dataset if isinstance(dataset, Subset) else dataset):
        pred = model.predict(img.unsqueeze(0).to('cpu'))
        show_prd_vs_gt(img, msk, pred)
    return
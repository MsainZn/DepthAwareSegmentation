import torch
import torch.optim as optim
from Evaluate import eval_model
from torch.utils.data import DataLoader
from ConcatModels import BaseUNet
import wandb
from Config import wandb_cfg, param_cfg
# from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device
) -> float:
    """Train the model for one epoch and return the average loss."""
    model.to(device)
    model.train()
    running_loss = 0.0
    loss_fn.to(device)
    for input_chnls, labels in dataloader:
        # Zero the gradients
        optimizer.zero_grad()  
        
        # Move data to the correct device
        labels = labels.to(device)
        x = {k: v.to(device) for k, v in input_chnls.items()}
        
        # Forward pass
        outputs = model(x)

        # Compute the loss
        dpt = x['depth'] if 'depth' in x else None
        loss = loss_fn(outputs, labels, dpt)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track running loss
        running_loss += loss.item()

    return running_loss / len(dataloader)

def pass_to_wandb(run: wandb.sdk.wandb_run.Run, 
                  epoch: int, 
                  metric_trn: dict,
                  metric_tst: dict) -> None:
    
    # Define metric groups for proper visualization
    run.define_metric("epoch")
    run.define_metric("loss", step_metric="epoch")  # Loss group
    run.define_metric("balanced_iou", step_metric="epoch")  # IoU group
    run.define_metric("balanced_acc", step_metric="epoch")  # Accuracy group
    run.define_metric("iou_per_class", step_metric="epoch")  # IoU per class (Grouped together)
    run.define_metric("dice_per_class", step_metric="epoch")  # Dice per class (Grouped together)
    run.define_metric("acc_per_class", step_metric="epoch")  # Accuracy per class (Grouped together)

    # Logging dictionary
    log_data = {"epoch": epoch}

    # Loss (1 plot, train and test together)
    log_data.update({
        "loss/train": metric_trn['loss'],  
        "loss/test": metric_tst['loss']
    })

    # Balanced IoU (1 plot, train and test together)
    log_data.update({
        "balanced_iou/train": metric_trn['balanced_iou'],  
        "balanced_iou/test": metric_tst['balanced_iou']
    })

    # Balanced Accuracy (1 plot, train and test together)
    log_data.update({
        "balanced_acc/train": metric_trn['balanced_acc'],  
        "balanced_acc/test": metric_tst['balanced_acc']
    })

    # Group all class metrics in the same plot per metric type
    for i in range(len(metric_trn['iou_per_class'])):
        log_data[f"iou_per_class/class_{i}/train"] = metric_trn['iou_per_class'][i]
        log_data[f"iou_per_class/class_{i}/test"] = metric_tst['iou_per_class'][i]

    for i in range(len(metric_trn['dice_per_class'])):
        log_data[f"dice_per_class/class_{i}/train"] = metric_trn['dice_per_class'][i]
        log_data[f"dice_per_class/class_{i}/test"] = metric_tst['dice_per_class'][i]

    for i in range(len(metric_trn['acc_per_class'])):
        log_data[f"acc_per_class/class_{i}/train"] = metric_trn['acc_per_class'][i]
        log_data[f"acc_per_class/class_{i}/test"] = metric_tst['acc_per_class'][i]

    # Log everything together
    run.log(log_data)
    return

def train_model(
    model: BaseUNet,
    trn_loader: DataLoader,
    val_loader: DataLoader,
    tst_loader: DataLoader,
    device: torch.device=torch.device("cpu"),
    optimizer:torch.optim.Optimizer=None,
    loss_fn:torch.nn.modules.loss._Loss=None,
    log_pth:str=None,
    wb_cfg:wandb_cfg=None,
    mdl_cfg:param_cfg=None
) -> tuple[list[dict], list[dict], dict]:
    
    """Train the model and return the training and validation metrics."""
    logger = log_progress if log_pth else print_progress
    print(wb_cfg.project, wb_cfg.entity)

    run = wandb.init(project=wb_cfg.project, 
                     entity=wb_cfg.entity, 
                     config=dict(mdl_cfg),
                     tags=[model.mdl_lbl])
    
    trn_val_list, val_val_list = [], []

    # scheduler = ReduceLROnPlateau(optimizer, 
    #                             mode='min', 
    #                             factor=0.5, 
    #                             patience=10, 
    #                             min_lr=1e-6)
    # scheduler = CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6) 

    for epoch in range(mdl_cfg.num_epochs):
        # Train the model
        train_one_epoch(model, trn_loader, optimizer, loss_fn, device)
        
        # Evaluate the model
        metric_trn = eval_model(model, trn_loader, model.num_class, loss_fn, device)
        metric_val = eval_model(model, val_loader, model.num_class, loss_fn, device)
        
        # scheduler.step(metric_val["loss"])  # scheduler based on val_loss
        # scheduler.step()
        
        # Append the metrics
        trn_val_list.append({epoch+1: metric_trn})
        val_val_list.append({epoch+1: metric_val})
        # Log the metrics    
        logger(epoch, metric_trn, metric_val, log_pth)

        # Pass to wandb
        pass_to_wandb(run, epoch, metric_trn, metric_val)
    
    # Log the testset metrics
    tst_dict = eval_model(model, tst_loader, model.num_class, loss_fn, device)
    logger(epoch, tst_dict, None, log_pth, titles=('Testset', None))
    run.finish()
    
    return trn_val_list, val_val_list, tst_dict

def print_progress(epoch: int, trn_metrics:dict, tst_metrics:dict, 
                   name:str=None, titles:dict=('Trainset', 'Validation set')) -> None:
    """Print the progress of the training."""
    print(f"-------------- Epoch {int(epoch)+1} --------------")
    print(f"{titles[0]}...")
    print_metrics(trn_metrics)
    if tst_metrics:
        print(f"{titles[1]} set...")
        print_metrics(tst_metrics)
    return

def print_metrics(metrics: dict, name: str = None) -> None:
    """Print the metrics for the current epoch, handling both lists and scalars."""
    for key, value in metrics.items():
        if isinstance(value, list):  # Handle lists
            formatted_values = ", ".join(f"{v:.4f}" for v in value)
            print(f"{key}: [{formatted_values}] | ", end="")
        else:  # Handle single numbers
            print(f"{key}: {value:.4f} | ", end="")
    print('')
    return

def log_progress(epoch: int, trn_metrics:dict, tst_metrics:dict=None, 
                 file_pth:str='metrics.log', titles:dict=('Trainset', 'Validation set')) -> None:
    """Log the progress of the training."""
    if trn_metrics:
        with open(file_pth, "a") as f:
            f.write(f"-------------- Epoch {int(epoch)+1} --------------\n")
            f.write(f"{titles[0]}...\n")
            f.close()
        log_metrics(trn_metrics, file_pth)
    
    if tst_metrics:
        with open(file_pth, "a") as f:
            f.write(f"{titles[1]}...\n")
            f.close()
        log_metrics(tst_metrics, file_pth)
    return
        
def log_metrics(metrics: dict, file_pth: str = 'metrics.log') -> None:
    """Log the metrics for the current epoch, handling both lists and scalars."""
    with open(file_pth, "a") as f:
        for key, value in metrics.items():
            if isinstance(value, list):  # Handle lists
                formatted_values = ", ".join(f"{v:.4f}" for v in value)
                f.write(f"{key}: [{formatted_values}] | ")
            else:  # Handle single numbers
                f.write(f"{key}: {value:.4f} | ")
        f.write("\n")
        f.close()
    return


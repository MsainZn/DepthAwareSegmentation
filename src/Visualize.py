import random
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np
from Preprocess import tensor_to_numpy_1D, tensor_to_numpy_rgb
from typing import Literal, Union
import os

def plot_avg_metrics(list_trn: list[int, dict], 
                     list_tst: list[int, dict],
                     save_dir: str=None,
                     name: str=None)-> None:
    # Initialize lists for each metric
    loss_trn, loss_tst = [], []
    balanced_iou_trn, balanced_iou_tst = [], []
    balanced_dice_trn, balanced_dice_tst = [], []
    balanced_acc_trn, balanced_acc_tst = [], []
    
    for trn, tst in zip(list_trn, list_tst):
        for (k_trn, v_trn), (k_tst, v_tst) in zip(trn.items(), tst.items()):
            # Collect values for each metric
            loss_trn.append(v_trn['loss'])
            loss_tst.append(v_tst['loss'])
            
            balanced_iou_trn.append(v_trn['balanced_iou'])
            balanced_iou_tst.append(v_tst['balanced_iou'])
            
            balanced_dice_trn.append(v_trn['balanced_dice'])
            balanced_dice_tst.append(v_tst['balanced_dice'])
            
            balanced_acc_trn.append(v_trn['balanced_acc'])
            balanced_acc_tst.append(v_tst['balanced_acc'])
    
    # Plot all metrics
    epochs = range(len(list_trn))

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, loss_trn, 'r', label='Train Loss')
    plt.plot(epochs, loss_tst, 'b', label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Balanced IoU
    plt.subplot(2, 2, 2)
    plt.plot(epochs, balanced_iou_trn, 'r', label='Train Balanced IoU')
    plt.plot(epochs, balanced_iou_tst, 'b', label='Validation Balanced IoU')
    plt.title('Balanced IoU')
    plt.xlabel('Epoch')
    plt.ylabel('Balanced IoU')
    plt.legend()

    # Plot Balanced Dice
    plt.subplot(2, 2, 3)
    plt.plot(epochs, balanced_dice_trn, 'r', label='Train Balanced Dice')
    plt.plot(epochs, balanced_dice_tst, 'b', label='Validation Balanced Dice')
    plt.title('Balanced Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Balanced Dice')
    plt.legend()

    # Plot Balanced Accuracy
    plt.subplot(2, 2, 4)
    plt.plot(epochs, balanced_acc_trn, 'r', label='Train Balanced Accuracy')
    plt.plot(epochs, balanced_acc_tst, 'b', label='Validation Balanced Accuracy')
    plt.title('Balanced Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Balanced Accuracy')
    plt.legend()

    plt.tight_layout()

    if save_dir is not None and name is not None:
        save_path = os.path.join(save_dir, f'{name}_metrics_avg.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save figure
    else:
        plt.show()
    return

def plot_metric_per_class(list_trn: list[dict], 
                          list_tst: list[dict], 
                          save_dir: str=None,
                          name: str=None)-> None:
    # Initialize lists for each metric
    iou_per_class_trn, iou_per_class_tst = [], []
    dice_per_class_trn, dice_per_class_tst = [], []
    acc_per_class_trn, acc_per_class_tst = [], []
    
    for trn, tst in zip(list_trn, list_tst):
        for (k_trn, v_trn), (k_tst, v_tst) in zip(trn.items(), tst.items()):
            # Collect per-class values
            iou_per_class_trn.append(v_trn['iou_per_class'])
            iou_per_class_tst.append(v_tst['iou_per_class'])
            
            dice_per_class_trn.append(v_trn['dice_per_class'])
            dice_per_class_tst.append(v_tst['dice_per_class'])
            
            acc_per_class_trn.append(v_trn['acc_per_class'])
            acc_per_class_tst.append(v_tst['acc_per_class'])
    
    # Plot all metrics
    epochs = range(len(list_trn))

    # Plot Loss
    plt.figure(figsize=(18, 5))
    colors = plt.cm.tab10.colors
    # IoU per Class plot (for the first class)
    plt.subplot(1, 3, 1)
    for class_idx in range(len(iou_per_class_trn[0])):  # assuming same length for each class
        iou_class_trn = [x[class_idx] for x in iou_per_class_trn]
        iou_class_tst = [x[class_idx] for x in iou_per_class_tst]
        clr = colors[class_idx % len(colors)]
        plt.plot(epochs, iou_class_trn, label=f'Class {class_idx} (Train)', linestyle='-', color=clr)
        plt.plot(epochs, iou_class_tst, label=f'Class {class_idx} (Validation)', linestyle='--', color=clr)
    plt.title('IoU per Class')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')

    # Dice per Class plot (for the first class)
    plt.subplot(1, 3, 2)
    for class_idx in range(len(dice_per_class_trn[0])):  # assuming same length for each class
        dice_class_trn = [x[class_idx] for x in dice_per_class_trn]
        dice_class_tst = [x[class_idx] for x in dice_per_class_tst]
        clr = colors[class_idx % len(colors)]
        plt.plot(epochs, dice_class_trn, label=f'Class {class_idx} (Train)', linestyle='-', color=clr)
        plt.plot(epochs, dice_class_tst, label=f'Class {class_idx} (Validation)', linestyle='--', color=clr)
    plt.title('Dice per Class')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')

    # Accuracy per Class plot (for the first class)
    plt.subplot(1, 3, 3)
    for class_idx in range(len(acc_per_class_trn[0])):  # assuming same length for each class
        acc_class_trn = [x[class_idx] for x in acc_per_class_trn]
        acc_class_tst = [x[class_idx] for x in acc_per_class_tst]
        clr = colors[class_idx % len(colors)]
        plt.plot(epochs, acc_class_trn, label=f'Class {class_idx} (Train)', linestyle='-', color=clr)
        plt.plot(epochs, acc_class_tst, label=f'Class {class_idx} (Validation)', linestyle='--', color=clr)
    plt.title('Accuracy per Class')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    # Move the legend outside the plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 0.75))

    plt.tight_layout()
        
    if save_dir is not None and name is not None:
        save_path = os.path.join(save_dir, f'{name}_metrics_per_class.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save figure
    else:
        plt.show()
    return

def plot_one_metric(list_trn:list[int, dict], 
                    list_tst:list[int, dict],
                    metric:str='loss')-> None:
    ls, lss = [], []
    for x, y in zip(list_trn, list_tst):
        for (k, v), (i, j) in zip(x.items(), y.items()):
            ls.append(v[metric])
            lss.append(j[metric])
    plt.plot(range(len(list_trn)), ls, 'r', label='train')
    plt.plot(range(len(list_trn)), lss, 'b', label='validation')
    
    plt.title(metric)
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.legend()
    plt.show()
    return

def show_pred(img_dict: dict[str: torch.Tensor], 
              prd_msk: torch.Tensor,
              save_dir: str=None,
              name: str=None)-> None:
        
    rgb_img_np = tensor_to_numpy_rgb(img_dict['rgb'])
    msk_prd_np = tensor_to_numpy_1D(prd_msk)

    _, axes = plt.subplots(1, 2, figsize=(6, 3))
    
    axes[0].imshow(rgb_img_np)
    axes[0].set_title('Image')
    
    axes[1].imshow(msk_prd_np, cmap='gray')
    axes[1].set_title('Pred-Mask')
    
    if save_dir is not None and name is not None:
        save_path = os.path.join(save_dir, f'{name}_pred_vs_gt.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save figure
    else:
        plt.show()
    return

def show_prd_vs_gt(img_dict: dict[str: torch.Tensor], 
                   gt_msk: torch.Tensor,
                   prd_msk: torch.Tensor,
                   save_dir: str=None,
                   name: str=None)-> None:
        
    rgb_img_np = tensor_to_numpy_rgb(img_dict['rgb'])
    msk_gt_np  = tensor_to_numpy_1D(gt_msk)
    msk_prd_np = tensor_to_numpy_1D(prd_msk)

    _, axes = plt.subplots(1, 3, figsize=(9, 3))
    
    axes[0].imshow(rgb_img_np)
    axes[0].set_title('Image')
    
    axes[1].imshow(msk_gt_np, cmap='gray')
    axes[1].set_title('GT-Mask')
    
    axes[2].imshow(msk_prd_np, cmap='gray')
    axes[2].set_title('Pred-Mask')
    
    if save_dir is not None and name is not None:
        save_path = os.path.join(save_dir, f'{name}_pred_vs_gt.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save figure
    else:
        plt.show()
    return

def show_msk_overlay(img_dict: dict[str: torch.Tensor], 
                     msk_tnsr: torch.Tensor, 
                     trg_clss: list=None,
                     figsize: tuple=(3,3),
                     cmap: str='gray') -> np.ndarray:

    img_np = tensor_to_numpy_rgb(img_dict['rgb'])
    msk_np = msk_tnsr.squeeze(0).cpu().numpy()

    masked_image = np.zeros_like(img_np)
    for target_class in trg_clss:
        masked_image[msk_np == target_class] = img_np[msk_np == target_class]

    # Display result
    show_image(masked_image, title="Segmentation Mask Overlay", figsize=figsize, cmap=cmap)
    return

def show_msk(msk_img: torch.Tensor, 
             title: str='Mask', 
             figsize: tuple=(3,3), cmap: str='gray')-> None:
    # Convert to numpy and clip values if necessary
    img = tensor_to_numpy_1D(msk_img)
    show_image(img, title, figsize, cmap)
    return

def show_rgb(img_dict: dict[str: torch.Tensor], 
             title: str='Image', 
             figsize: tuple=(3,3), 
             cmap: str=None)-> None:
    # Convert from (C, H, W) -> (H, W, C) for matplotlib
    img = tensor_to_numpy_rgb(img_dict['rgb'])
    show_image(img, title, figsize, cmap)
    return

def show_image(img:np.ndarray, 
               title: str='Image', 
               figsize: tuple=(3,3), 
               cmap:str=None)-> None:
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.show()
    return

def show_pairs(img_dict: dict[str: torch.Tensor], 
               msk_tnsr: torch.Tensor, 
               figsize=(6, 3))-> None:
    
    rgb_tensor = img_dict['rgb']

    rgb_img = tensor_to_numpy_rgb(rgb_tensor)
    msk_img = tensor_to_numpy_1D(msk_tnsr)

    _, axes = plt.subplots(1, 2, figsize=figsize)

    # Ensure the axes have the same aspect ratio
    for ax in axes:
        ax.set_aspect('equal')  # Forces equal scaling of x and y dimensions
    
    axes[0].imshow(rgb_img)
    axes[0].set_title('Image')
    
    axes[1].imshow(msk_img, cmap='gray')
    axes[1].set_title('Mask')
    
    plt.show()
    return

def show_pairs_random(dataset:Dataset)-> None:
    id = random.randint(0, len(dataset)-1)
    image, mask = dataset[id]
    show_pairs(image, mask)
    return


def show_channel(img_dict: dict[str: torch.Tensor],
                 chnl: Literal['rgb', 'grey', 'depth', 'hog', 'hhaar', 'vhaar', 'dhaar'],
                 figsize: tuple=(3, 3))-> None:

    show_rgb(img_dict['rgb'], figsize=figsize) if chnl == 'rgb' \
        else show_msk(img_dict[chnl], figsize=figsize)

    return

def show_all_channels(input: Union[Dataset, tuple[dict[str, torch.Tensor], torch.Tensor]], 
                      figsize: tuple = (12, 6))-> None:  
    # Get image and mask from a tuple
    if isinstance(input, tuple):
        img_dict, msk_tnsr = input

    # Get a random image from the dataset
    elif isinstance(input, Dataset):
        idx = random.randint(0, len(input) - 1)
        img_dict, msk_tnsr = input[idx]
    else:
        raise ValueError("Input must be of type Dataset or a tuple of (image, mask)")
        
    chnl_names = list(img_dict.keys())  # Get channel names
    chnl_count = len(chnl_names) + 1  # +1 for the mask

    # Define the subplot layout dynamically
    cols = 4
    rows = (chnl_count // cols) + (chnl_count % cols > 0)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()  # Flatten for easier indexing

    # Plot each channel dynamically
    for i, key in enumerate(chnl_names):
        axes[i].imshow(tensor_to_numpy_rgb(img_dict[key])) if key == 'rgb' \
            else axes[i].imshow(tensor_to_numpy_1D(img_dict[key]), cmap='gray')
        axes[i].set_title(key) 

    # Plot Maks
    axes[len(chnl_names)].imshow(tensor_to_numpy_1D(msk_tnsr), cmap='gray')
    axes[len(chnl_names)].set_title('Mask')

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

    return
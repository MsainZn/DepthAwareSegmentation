import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as DS
from Config import param_cfg, path_cfg, dataset_cfg, typecheck_cfg
from Dataset import split_dataset, SegmentationDataset, collate_fn

def generate_metrics_table(models_data, save_dir=None, save_format="txt"):
    """
    Generates a tabular representation of model metrics and saves it as a text file.
    
    :param models_data: Dictionary where keys are model names, and values contain train, val, test metrics.
    :param save: Whether to save the results. If True, the table will be saved in the save_dir.
    :param save_dir: Directory to save the table if save=True.
    :param save_format: Format to save the table (e.g., 'txt').
    """
    model_names = list(models_data.keys())
    overall_metrics = ['balanced_acc', 'balanced_iou', 'balanced_dice', 'loss']
    class_metrics = ['acc_per_class', 'iou_per_class', 'dice_per_class']
    num_classes = len(next(iter(models_data.values()))['train']['iou_per_class'])
    
    result_str = """Overall Metrics:
    -----------------------------------------------------------------------------
    Model Name       | Train Balanced Acc | Val Balanced Acc | Test Balanced Acc 
                    | Train Balanced IoU | Val Balanced IoU | Test Balanced IoU 
                    | Train Balanced Dice | Val Balanced Dice | Test Balanced Dice 
                    | Train Loss | Val Loss | Test Loss 
    -----------------------------------------------------------------------------
    """
    
    for model in model_names:
        result_str += f"{model:<15}"
        for metric in overall_metrics:
            result_str += f"| {models_data[model]['train'][metric]:<18.4f} "
            result_str += f"| {models_data[model]['val'][metric]:<18.4f} "
            result_str += f"| {models_data[model]['test'][metric]:<18.4f} \n"
        result_str += "-----------------------------------------------------------------------------"
    
    # Per-class metrics
    for class_idx in range(num_classes):
        result_str += f"\nClass {class_idx + 1} Metrics:\n"
        result_str += "-----------------------------------------------------------------------------"
        result_str += "Model Name       | Train Acc | Val Acc | Test Acc | Train IoU | Val IoU | Test IoU | Train Dice | Val Dice | Test Dice\n"
        result_str += "-----------------------------------------------------------------------------"
        for model in model_names:
            result_str += f"{model:<15}"
            for metric in class_metrics:
                result_str += f"| {models_data[model]['train'][metric][class_idx]:<10.4f} "
                result_str += f"| {models_data[model]['val'][metric][class_idx]:<10.4f} "
                result_str += f"| {models_data[model]['test'][metric][class_idx]:<10.4f} "
            result_str += "\n"
        result_str += "-----------------------------------------------------------------------------"
    
    if save_dir:
        file_path = os.path.join(save_dir, f'metrics_table.{save_format}')
        with open(file_path, "w") as f:
            f.write(result_str)
    
    return result_str

def plot_model_metrics(models_data, save_dir=None, save_format="png"):
    """
    Plots histograms for given model metrics, including per-class metrics, with an option to save the plots.
    
    :param models_data: Dictionary where keys are model names, and values contain train, val, test metrics.
    :param save: Whether to save the plots. If True, plots will be saved in the save_dir.
    :param save_dir: Directory to save the plots if save=True.
    :param save_format: Format to save the plots (e.g., 'png', 'jpg').
    """
    model_names = models_data.keys()
    metric_keys = ['balanced_iou', 'balanced_dice', 'balanced_acc', 'loss', 'iou_per_class', 'dice_per_class', 'acc_per_class']

    num_models = len(model_names)
    width = 0.25  # Bar width

    for metric in metric_keys:
        if 'per_class' in metric:
            # If the metric is per-class, plot each class separately
            num_classes = len(next(iter(models_data.values()))['train'][metric])  # Extract the number of classes
            
            for class_idx in range(num_classes):
                plt.figure(figsize=(10, 5))  # Increase the figure size further
                x = np.arange(num_models)  # X locations for bars
                
                train_values = [models_data[model]['train'][metric][class_idx] for model in model_names]
                val_values = [models_data[model]['val'][metric][class_idx] for model in model_names]
                test_values = [models_data[model]['test'][metric][class_idx] for model in model_names]

                plt.bar(x - width, train_values, width, label='Train', color='b', alpha=0.7)
                plt.bar(x, val_values, width, label='Validation', color='g', alpha=0.7)
                plt.bar(x + width, test_values, width, label='Test', color='r', alpha=0.7)

                plt.xticks(ticks=x, labels=model_names, rotation=90, ha='right', fontsize=8)  # Reduce font size if needed
                plt.ylabel(f'{metric.replace("_", " ").title()}')
                plt.title(f'{metric.replace("_", " ").title()} - Class {class_idx + 1}')
                plt.legend()
                plt.grid(axis='y', linestyle='--', alpha=0.6)

                # Adjust layout with more space at the bottom
                plt.subplots_adjust(bottom=0.5)

                # Save the figure if the save option is enabled
                if save_dir:
                    plt.savefig(os.path.join(save_dir, f'{metric}_class_{class_idx + 1}.{save_format}'), format=save_format)
                else:
                    plt.show()
                plt.close()

        else:
            # If the metric is a scalar, plot a grouped bar chart for train, val, and test
            plt.figure(figsize=(10, 5))  # Increase the figure size further
            x = np.arange(num_models)

            train_values = [models_data[model]['train'][metric] for model in model_names]
            val_values = [models_data[model]['val'][metric] for model in model_names]
            test_values = [models_data[model]['test'][metric] for model in model_names]

            plt.bar(x - width, train_values, width, label='Train', color='b', alpha=0.7)
            plt.bar(x, val_values, width, label='Validation', color='g', alpha=0.7)
            plt.bar(x + width, test_values, width, label='Test', color='r', alpha=0.7)

            plt.xticks(ticks=x, labels=model_names, rotation=90, ha='right', fontsize=8)  # Reduce font size if needed
            plt.ylabel(metric.replace('_', ' ').title())
            plt.title(f'{metric.replace("_", " ").title()} Across Models')
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.6)

            # Adjust layout with more space at the bottom
            plt.subplots_adjust(bottom=0.5)

            # Save the figure if the save option is enabled
            if save_dir:
                plt.savefig(os.path.join(save_dir, f'{metric}.{save_format}'), format=save_format)
            else:
                plt.show()
            plt.close()

def plot_model_metrics_subplots_yvar(models_data, save_dir=None, save_format="png", y_limits=None):
    
    """
    Plots grouped metrics in subplots and allows saving the plots.
    
    :param models_data: Dictionary where keys are model names, and values contain train, val, test metrics.
    :param save: Whether to save the plots. If True, plots will be saved in the save_dir.
    :param save_dir: Directory to save the plots if save=True.
    :param save_format: Format to save the plots (e.g., 'png', 'jpg').
    :param y_limits: Dictionary specifying y-axis limits for metrics, e.g., {'balanced_acc': (0, 1)}
    """
    default_y_limits = {
            'balanced_acc': (0.8, 1.05),
            'balanced_iou': (0.8, 1.05),
            'balanced_dice': (0.8, 1.05),
            'loss': (0, 1.05),
            'acc_per_class': (0.8, 1.05),
            'iou_per_class': (0.6, 1.05),
            'dice_per_class': (0.5, 1.05)
        }
    
    if y_limits is None:
        y_limits = default_y_limits
        

    model_names = list(models_data.keys())
    num_models = len(model_names)
    width = 0.25  # Bar width
    
    # --- Subplot for overall metrics ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    overall_metrics = ['balanced_acc', 'balanced_iou', 'balanced_dice', 'loss']
    axes = axes.flatten()
    
    for i, metric in enumerate(overall_metrics):
        x = np.arange(num_models)
        train_values = [models_data[m]['train'][metric] for m in model_names]
        val_values = [models_data[m]['val'][metric] for m in model_names]
        test_values = [models_data[m]['test'][metric] for m in model_names]
        
        axes[i].bar(x - width, train_values, width, label='Train', color='b', alpha=0.7)
        axes[i].bar(x, val_values, width, label='Validation', color='g', alpha=0.7)
        axes[i].bar(x + width, test_values, width, label='Test', color='r', alpha=0.7)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(model_names, rotation=90, fontsize=8)
        axes[i].set_title(metric.replace('_', ' ').title())
        axes[i].grid(axis='y', linestyle='--', alpha=0.6)
        axes[i].legend()
        
        if y_limits and metric in y_limits:
            axes[i].set_ylim(y_limits[metric])
    
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f'overall_metrics.{save_format}'), format=save_format)
    else:
        plt.show()
    plt.close()
    
    # --- Subplots for per-class metrics ---
    num_classes = len(next(iter(models_data.values()))['train']['iou_per_class'])
    for class_idx in range(num_classes):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        class_metrics = ['acc_per_class', 'iou_per_class', 'dice_per_class']
        
        for i, metric in enumerate(class_metrics):
            x = np.arange(num_models)
            train_values = [models_data[m]['train'][metric][class_idx] for m in model_names]
            val_values = [models_data[m]['val'][metric][class_idx] for m in model_names]
            test_values = [models_data[m]['test'][metric][class_idx] for m in model_names]
            
            axes[i].bar(x - width, train_values, width, label='Train', color='b', alpha=0.7)
            axes[i].bar(x, val_values, width, label='Validation', color='g', alpha=0.7)
            axes[i].bar(x + width, test_values, width, label='Test', color='r', alpha=0.7)
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(model_names, rotation=90, fontsize=8)
            axes[i].set_title(f'Class {class_idx + 1} - {metric.replace("_", " ").title()}')
            axes[i].grid(axis='y', linestyle='--', alpha=0.6)
            axes[i].legend()
            
            if y_limits and metric in y_limits:
                axes[i].set_ylim(y_limits[metric])
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, f'class_{class_idx + 1}_metrics.{save_format}'), format=save_format)
        else:
            plt.show()
        plt.close()

def validate_config_model(config_models:dict[str, param_cfg], 
                          config_checks:typecheck_cfg):
    # Config Asserts:: Check Models with Invalid Model Type or Loss Func
    for model_name, model_cfg in config_models.items():
        # Check Model Type
        assert model_cfg.mdl_type in config_checks.valid_models, \
            f"Invalid model type for {model_name}: {model_cfg.mdl_type}. Must be one of {config_checks.valid_models}"
    return

def validate_config_chnls(config_models:dict[str, param_cfg], 
                          config_checks:typecheck_cfg):
    
    for model_name, model_cfg in config_models.items():
        # Check all the channels in the chanel list are in valid channel list
        for chnl in model_cfg.in_channels:
            assert chnl in config_checks.valid_channels, \
                f"Invalid channel {chnl} for {model_name}. Must be one of {config_checks.valid_channels}"
    return

def validate_config_loss(config_models:dict[str, param_cfg], 
                         config_checks:typecheck_cfg):
    # Config Asserts:: Check Models with Invalid Model Type or Loss Func
    for model_name, model_cfg in config_models.items():
        assert model_cfg.loss_fn in config_checks.valid_loss_fn_nrm + config_checks.valid_loss_fn_ord, \
            f"Invalid loss function for {model_name}: {model_cfg.loss_fn}. Must be one of {config_checks.valid_loss_fn_nrm + config_checks.valid_loss_fn_ord}"
    return

def manage_dataset(mdl_conf:param_cfg, 
                   path_conf:path_cfg, 
                   ds_conf:dataset_cfg
    )->tuple[DS, DS, DS, SegmentationDataset]:
    
    # Base fix for config
    in_channel_count = len(mdl_conf.in_channels) + 2 if 'rgb' in mdl_conf.in_channels else len(mdl_conf.in_channels)
    name = f'{ds_conf.dta_name}_{in_channel_count}x{mdl_conf.img_size}x{mdl_conf.img_size}-{"_".join(mdl_conf.in_channels)}'
    
    # DS based on config file
    dataset_path = os.path.join(path_conf.opath.DTA_BIN_DIR, f'{name}.pkl')

    # Load or Construct Dataset
    if os.path.exists(dataset_path):
        dataset = SegmentationDataset().load_from_file(path_conf.opath.DTA_BIN_DIR, name)
    else:
        dataset = SegmentationDataset().set_save_dir(path_conf.opath.DTA_BIN_DIR).\
                set_process_dirs(path_conf.ipath.IMG_DTA_DIR, path_conf.ipath.MSK_DTA_DIR).\
                set_channels(mdl_conf.in_channels).set_img_ext(list(ds_conf.img_file_ext)).\
                set_msk_ext(list(ds_conf.msk_file_ext)).load_ds_to_mem(mdl_conf.img_size).\
                save_to_file()
        
    # Split dataset into training, testing and validation
    trn_ds, val_ds, tst_ds = split_dataset(
                        dataset=dataset, 
                        train_size=ds_conf.split_ratio.train,
                        val_size=ds_conf.split_ratio.val,
                        test_size=ds_conf.split_ratio.test,
                        seed=ds_conf.split_ratio.random_seed)
        
    return trn_ds, val_ds, tst_ds, dataset

def manage_dataloader(trn_ds:SegmentationDataset,
                      val_ds:SegmentationDataset,
                      tst_ds:SegmentationDataset,
                      ds_conf:dataset_cfg,
                      collate_fn:callable=collate_fn
                      ) -> tuple[DataLoader]:

    # Create DataLoaders for training and testing
    trn_dl = DataLoader(dataset=trn_ds, 
                        batch_size=ds_conf.batch_size, 
                        shuffle=True,  
                        num_workers=ds_conf.num_workers,
                        collate_fn=collate_fn)
    
    tst_dl = DataLoader(dataset=tst_ds, 
                        batch_size=ds_conf.batch_size, 
                        shuffle=False, 
                        num_workers=ds_conf.num_workers,
                        collate_fn=collate_fn)

    val_dl = DataLoader(dataset=val_ds, 
                        batch_size=ds_conf.batch_size, 
                        shuffle=False, 
                        num_workers=ds_conf.num_workers,
                        collate_fn=collate_fn)

    return trn_dl, val_dl, tst_dl
    
def validate_config_path(path_config:path_cfg):
    
    # Input Paths
    assert os.path.exists(path_config.ipath.CFG_DIR),     'Config folder does not exist'
    assert os.path.exists(path_config.ipath.IMG_DTA_DIR), 'IMG folder does not exist'
    assert os.path.exists(path_config.ipath.MSK_DTA_DIR), 'MSK folder does not exist'
    # Output Paths
    assert os.path.exists(path_config.opath.MDL_LOG_DIR), 'Logs folder does not exist' 
    assert os.path.exists(path_config.opath.RES_BIN_DIR), 'Results folder does not exist' 
    assert os.path.exists(path_config.opath.MDL_BIN_DIR), 'Models folder does not exist' 
    assert os.path.exists(path_config.opath.GRF_PLT_DIR), 'Plot folder does not exist' 
    assert os.path.exists(path_config.opath.SMP_PLT_DIR), 'Sample folder does not exist'
    assert os.path.exists(path_config.opath.SLR_LOG_DIR), 'SLURM folder does not exist'
    assert os.path.exists(path_config.opath.DTA_BIN_DIR), 'Data folder does not exist'
    
    return

def validate_dataset_name(name:str, config_checks:typecheck_cfg):
    assert name in config_checks.valid_dataset_names, \
        f"Invalid Name: {name} for dataset. Must be one of {config_checks.valid_dataset_names}"
    return
    

def manage_artifact_dir(cfg:path_cfg) -> None:
    """Creates output folders from Hydra config."""
    prj_dir = os.getenv("PRJ_DIR", cfg.get("PRJ_DIR", "."))
    # Rename Previous Artical Folder as ART_DIR_old
    if os.path.exists(cfg.opath.ART_DIR):
        # add datetime to the folder name
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        older_dir_sfx = cfg.opath.ART_DIR + '_' + datetime_str
        # rename the folder
        os.rename(cfg.opath.ART_DIR, older_dir_sfx)
        # Now move this directory to the History Folder
        shutil.move(older_dir_sfx, os.path.join(prj_dir, cfg.ipath.HIST_DIR))
        print(f"⚠️ You can find you previous experiment under -> {older_dir_sfx} in {cfg.ipath.HIST_DIR} directory!")
    # Create new ART_DIR
    for path in cfg.opath.values():
        os.makedirs(os.path.join(prj_dir, path), exist_ok=True)
        # add gitkeep in all folders
        gitkeep_path = os.path.join(prj_dir, path, ".gitkeep")
        if not os.path.exists(gitkeep_path):
            with open(gitkeep_path, "w") as f:
                f.write(" ")
        # print(f"✅ Created: {os.path.join(prj_dir, path)}")
    return

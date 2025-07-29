from dotenv import load_dotenv
from hydra.core.config_store import ConfigStore
import hydra
from ConcatModels import BaseUNet, Model_Factory
from LossFunctions import Loss_Factory, BaseLoss
from Trainer import manage_dataset, manage_dataloader, manage_artifact_dir,\
validate_config_path, validate_config_loss, validate_config_model, validate_config_chnls, validate_dataset_name, \
plot_model_metrics_subplots_yvar, plot_model_metrics, generate_metrics_table

from Preprocess import set_cuda_reproducible

import os
import torch
from torch.optim import Adam
from Visualize import plot_avg_metrics, show_prd_vs_gt, plot_metric_per_class
import pickle
from Evaluate import eval_model
from Train import train_model
from Postprocess import save_checkpoint
from Dataset import add_btch_dim

# Register Config Type
from Config import setup_cfg
cs = ConfigStore.instance()
cs.store(name="config", node=setup_cfg)

@hydra.main(version_base=None, config_path="../configs", config_name="main_config")
def main(config: setup_cfg) -> int:
    
    # Load environment variables
    set_cuda_reproducible(42)
    load_dotenv()

    # Create Necessary Paths
    if config.auto_manage_artifacts:
        manage_artifact_dir(config.path)

    # Validate Configs 
    validate_config_path(config.path)
    validate_config_model(config.models, config.typecheck)
    validate_config_chnls(config.models, config.typecheck)
    validate_config_loss(config.models,  config.typecheck)
    validate_dataset_name(config.dataset.dta_name,  config.typecheck)

    # Main Loop
    res_dict_total = {}
    for model_name, model_cfg in config.models.items():
        
        # Create Model Specific Dataset Or Assign if already Exist
        print('✅ Managing Dataset')
        trn_ds, val_ds, tst_ds, dataset = manage_dataset(model_cfg, config.path, config.dataset)
        trn_dl, val_dl, tst_dl = manage_dataloader(trn_ds, val_ds, tst_ds, config.dataset)
        
        # Instantiating Seg Models
        print('✅ Initializing Model')
        model:BaseUNet = Model_Factory.create_model(
                input_mdl = model_cfg.mdl_type,
                in_channel_count = dataset.in_channel_count, 
                num_class = dataset.num_class,
                is_ordinal = True if model_cfg.loss_fn in config.typecheck.valid_loss_fn_ord else False,
                mdl_lbl = model_name,
                img_size = dataset.img_size)

        optimizer = Adam(model.parameters(), 
                         lr=model_cfg.lr, 
                         weight_decay=model_cfg.weight_decay)
        
        # Assign Loss Function
        print('✅ Initializing Loss Function')
        loss_fn:BaseLoss = Loss_Factory.create_loss(
            loss_type=model_cfg.loss_fn, 
            weights=dataset.class_weights,
            device=config.dev_type,
            clses=model_cfg.clses if model_cfg.clses is not None else None,
        )

        # Update log path and delete old logs
        log_pth = os.path.join(config.path.opath.MDL_LOG_DIR, f'{model_name}.log')
        
        # Train models
        print('✅ Training Model')
        list_trn, list_val, tst_dict = train_model(
            model=model,
            trn_loader=trn_dl,
            val_loader=val_dl,
            tst_loader=tst_dl,
            device=torch.device(config.dev_type),
            optimizer=optimizer,
            loss_fn=loss_fn,
            log_pth=log_pth,
            wb_cfg=config.wandb,
            mdl_cfg=model_cfg,
        )
        
        # Save training and validation metrics
        print('✅ Eval Model')
        res_dict_total[model_name] = {
            'train': eval_model(model, trn_dl, model.num_class, loss_fn, torch.device(config.dev_type)),
            'val':   eval_model(model, val_dl, model.num_class, loss_fn, torch.device(config.dev_type)),
            'test':  eval_model(model, tst_dl, model.num_class, loss_fn, torch.device(config.dev_type))
        }

        # plot metrics
        print('✅ Plot Metrics')
        plot_avg_metrics(list_trn, list_val, config.path.opath.GRF_PLT_DIR, model_name)
        plot_metric_per_class(list_trn, list_val, config.path.opath.GRF_PLT_DIR, model_name)
        
        # save models results as pickle
        pickle_path = os.path.join(config.path.opath.RES_BIN_DIR, f'{model_name}_metrics.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump({'train': list_trn, 'val': list_val, 'test': tst_dict}, f)
        
        # save model
        print('✅ Save Model')
        save_checkpoint(model, model_name, config.path.opath.MDL_BIN_DIR, optimizer, model_cfg.num_epochs)

        # visual prediction model 
        print('✅ Visual Prediction')  
        img_dir = os.path.join(config.path.opath.SMP_PLT_DIR, model_name)
        os.makedirs(img_dir, exist_ok=True) 
        for i, (img_dict, msk)in enumerate(tst_ds):
            pred = model.predict_one(add_btch_dim(img_dict))
            show_prd_vs_gt(img_dict, msk, pred, img_dir, f'{model_name}_tst_img_{i}')

        for i, (img_dict, msk) in enumerate(val_ds):
            pred = model.predict_one(add_btch_dim(img_dict))
            show_prd_vs_gt(img_dict, msk, pred, img_dir, f'{model_name}_val_img_{i}')
        
        print('✅ Done for model:', model_name)

    # Comparison of models in the final Post Process
    plot_model_metrics_subplots_yvar(res_dict_total, save_dir=config.path.opath.GRF_PLT_DIR)
    plot_model_metrics(res_dict_total, save_dir=config.path.opath.GRF_PLT_DIR)
    generate_metrics_table(res_dict_total, save_dir=config.path.opath.GRF_PLT_DIR)

    return 0

if __name__ == "__main__":
    main()






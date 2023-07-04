# Copyright 2023 Motorica AB, Inc. All Rights Reserved.

import os
import sys

import numpy as np
from utils.motion_dataset import MotionDataset
from models.LightningModel import LitLDA
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelPruning, QuantizationAwareTraining
from utils.hparams import get_hparams
from pathlib import Path
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 553335
seed_everything(RANDOM_SEED)

def data_loader(dataset_root, file_name, data_hparams, batch_size, num_workers=16, shuffle=True):

    print("dataset_root: " + dataset_root)
    dataset = MotionDataset(
        dataset_root,
        Path(dataset_root) / file_name,
        data_hparams=data_hparams,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=shuffle,
        drop_last=True,
    )

def dataloaders(dataset_root, data_hparams, batch_size, num_workers):

    train_dl = data_loader(dataset_root, data_hparams["traindata_filename"], hparams.Data, batch_size, num_workers, shuffle=True)    
    val_dl = data_loader(dataset_root, data_hparams["testdata_filename"], hparams.Data, batch_size, num_workers, shuffle=True)
    test_dl = data_loader(dataset_root, data_hparams["testdata_filename"], hparams.Data, batch_size, num_workers, shuffle=False)
    
    return train_dl, val_dl, test_dl

if __name__ == "__main__":

    hparams, conf_name = get_hparams()
    assert os.path.exists(
        hparams.dataset_root
    ), "Failed to find root dir `{}` of dataset.".format(hparams.dataset_root)

    train_dl, val_dl, test_dl = dataloaders(hparams.dataset_root, hparams.Data, hparams.batch_size, hparams.num_dataloader_workers)
    
    if hparams.Trainer["resume_from_checkpoint"] is not None:
        # Load model for finetuning or resuming training
        ckpt=hparams.Trainer["resume_from_checkpoint"]
        print(f"resuming from checkpoint: {ckpt}")
        model = LitLDA.load_from_checkpoint(ckpt, cfg=hparams)
        print("Reusing the scalers from previous model.")
        scalers = model.get_scalers()
    else:
    
        # Create new model
        print("Fitting scalers")
        scalers = train_dl.dataset.fit_scalers()
        
        print("Setting scalers to model hparams")
        hparams.Data["scalers"] = scalers
        
        print("create model")
        model = LitLDA(hparams)
        
    
    # Standardize data
    print("standardize data")
    train_dl.dataset.standardize(scalers)
    val_dl.dataset.standardize(scalers)
    test_dl.dataset.standardize(scalers)

    trainer_params = vars(hparams).copy()

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        verbose=True,
        save_top_k=4,
        monitor='Loss/train',
        mode='min'
    )
    
    callbacks = [lr_monitor, checkpoint_callback]
    trainer = Trainer(callbacks=callbacks,**(trainer_params["Trainer"]))
    
    print("Start training!")
    trainer.fit(model, train_dl, val_dl)


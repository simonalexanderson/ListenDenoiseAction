# Copyright 2023 Motorica AB, Inc. All Rights Reserved.

import os

import torch
import numpy as np
from torch.optim import SGD, Adam, RMSprop
from torch.optim.lr_scheduler import LambdaLR, MultiplicativeLR, StepLR, CosineAnnealingWarmRestarts
from pytorch_lightning import LightningModule
from utils.logging_mixin import LoggingMixin

from typing import Tuple, Optional, Union, Dict
from argparse import Namespace

class BaseModel(LoggingMixin, LightningModule):
    def __init__(self, conf: Optional[Union[Dict, Namespace]] = None, **kwargs):
        super().__init__()
        
        self.save_hyperparameters(conf)
        
        scalers = self.hparams.Data["scalers"]
        #Get input and output scalers from hparams
        input_means=np.array([])
        input_stds=np.array([])
        if scalers["in_scaler"] is not None:
            input_means = scalers["in_scaler"].mean_
            input_stds = scalers["in_scaler"].scale_    

        self.input_means = torch.from_numpy(input_means)
        self.input_scales = torch.from_numpy(input_stds)
        self.output_means = torch.from_numpy(scalers["out_scaler"].mean_)
        self.output_scales = torch.from_numpy(scalers["out_scaler"].scale_)
        
    def get_scalers(self):
        return self.hparams.Data["scalers"]
        
    # standarize input
    def standardizeInput(self, input_tensor):
        return ((input_tensor - self.input_means.type_as(input_tensor)) / self.input_scales.type_as(input_tensor))

    # standarize output
    def standardizeOutput(self, output_tensor):
        return ((output_tensor - self.output_means.type_as(output_tensor)) / self.output_scales.type_as(output_tensor))

    # Add scale and means to output
    def destandardizeInput(self, input_tensor):
        return (input_tensor * self.input_scales.type_as(input_tensor) + self.input_means.type_as(input_tensor))

    # Add scale and means to output
    def destandardizeOutput(self, predictions):
        return (predictions * self.output_scales.type_as(predictions) + self.output_means.type_as(predictions))
        
    def configure_optimizers(self):
        lr_params = self.hparams.Optim
        optim_args = lr_params["args"][lr_params["name"]]
        optimizers = {"adam": Adam, "sgd": SGD, "rmsprop": RMSprop}
        # Define optimizer
        optimizer = optimizers[lr_params["name"]](
            self.parameters(), lr=self.hparams.lr, **optim_args
        )

        # Define Learning Rate Scheduling
        def lambda1(val):
            return lambda epoch: epoch // val

        sched_params = self.hparams.Optim["Schedule"]
        sched_name = sched_params["name"]
        if not sched_name:
            return optimizer

        sched_args = sched_params["args"][sched_name]

        if sched_name == "step":
            scheduler = StepLR(optimizer, **sched_args)
        elif sched_name == "multiplicative":
            scheduler = MultiplicativeLR(
                optimizer, lr_lambda=[lambda1(sched_args["val"])]
            )
        elif sched_name == "lambda":
            scheduler = LambdaLR(optimizer, lr_lambda=[lambda1(sched_args["val"])])
        elif sched_name == "cos_warm":
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=sched_args["T_0"])
        else:
            raise NotImplementedError("Unimplemented Scheduler!")

        #return [optimizer], [scheduler]
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    # learning rate warm-up
    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        lr = self.hparams.lr
        #warm up lr
        warm_up = self.hparams.Optim["Schedule"]["warm_up"]
        if self.trainer.global_step < warm_up:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / warm_up)
            lr *= lr_scale
            for pg in optimizer.param_groups:
                pg["lr"] = lr
        



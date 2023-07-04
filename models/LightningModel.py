# Copyright 2023 Motorica AB, Inc. All Rights Reserved.

import os
import sys

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from models.BaseModel import BaseModel
from models.nn import LDA

from typing import Tuple, Optional, Union, Dict
from argparse import Namespace

class LitLDA(BaseModel):
    def __init__(self, conf, **kwargs):
        super().__init__(conf)

        self.input_dim = 0
        self.style_dim = 0
        if self.hparams.Data["scalers"]["in_scaler"] is not None:
            self.input_dim = self.hparams.Data["scalers"]["in_scaler"].mean_.shape[0]
        if self.hparams.Data["scalers"]["style_scaler"] is not None:
            self.style_dim = self.hparams.Data["scalers"]["style_scaler"].mean_.shape[0]
        self.pose_dim = self.hparams.Data["scalers"]["out_scaler"].mean_.shape[0]
        self.g_cond_dim = self.style_dim
        self.unconditional = self.input_dim == 0
        n_timesteps = self.hparams.Data["segment_length"]
        diff_params = self.hparams.Diffusion
            
        beta_min = diff_params["noise_schedule_start"]
        beta_max = diff_params["noise_schedule_end"]
        self.n_noise_schedule = diff_params["n_noise_schedule"]
        
        self.noise_schedule_name = "linear"                                                         
        self.noise_schedule = torch.linspace(beta_min, beta_max, self.n_noise_schedule)
        self.noise_level = torch.cumprod(1 - self.noise_schedule, dim=0)
        
        nn_name = diff_params["name"]
        nn_args = diff_params["args"][nn_name]
        
        self.diffusion_model = LDA(self.pose_dim, 
                                        self.hparams.Diffusion["residual_layers"],
                                        self.hparams.Diffusion["residual_channels"],
                                        self.hparams.Diffusion["embedding_dim"],
                                        self.input_dim,
                                        self.g_cond_dim,
                                        self.n_noise_schedule,
                                        nn_name,
                                        nn_args)
                                        
        self.loss_fn = nn.MSELoss()
        
    
    def get_input_dim(self):
        return self.input_dim
        
    def get_style_dim(self):
        return self.style_dim
        
    def get_pose_dim(self):
        return self.pose_dim
        
    def diffusion(self, poses, t):
        N, T, C = poses.shape
        noise = torch.randn_like(poses)
        noise_scale = self.noise_level.type_as(noise)[t].unsqueeze(1).unsqueeze(2).repeat(1,T,C)
        noise_scale_sqrt = noise_scale**0.5
        noisy_poses = noise_scale_sqrt * poses + (1.0 - noise_scale)**0.5 * noise
        return noisy_poses, noise
        
    def forward(self, batch):

        ctrl, global_cond, poses =batch
                
        N, T, C = poses.shape

        num_noisesteps = self.n_noise_schedule
        t = torch.randint(0, num_noisesteps, [N], device=poses.device)

        noise = torch.randn_like(poses)
        noise_scale = self.noise_level.type_as(noise)[t].unsqueeze(1).unsqueeze(2).repeat(1,T,C)
        noise_scale_sqrt = noise_scale**0.5
        noisy_poses = noise_scale_sqrt * poses + (1.0 - noise_scale)**0.5 * noise
        noisy_poses, noise = self.diffusion(poses, t)
        predicted = self.diffusion_model(noisy_poses, ctrl, global_cond, t)

        loss = self.loss_fn(noise, predicted.squeeze(1))
                
        return loss

    def training_step(self, batch, batch_idx):
        loss = self(batch)
        self.log('Loss/train', loss, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):

        loss = self(batch)

        self.log('val_loss', loss, prog_bar=True, sync_dist=True)

        if (self.trainer.global_step > 0 and
            batch_idx==0 and
            (self.trainer.current_epoch == self.trainer.max_epochs-1 or self.trainer.current_epoch % self.hparams.Validation["render_every_n_epochs"]==0)):

            # Log results for the validation data
            self.synthesize_and_log(batch, "val")

        
        output = {"val_loss": loss}
        return output

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log('Loss/val', avg_loss, sync_dist=True)

    def synthesize_and_log(self, batch, log_prefix):
        ctrl, g_cond, _ =batch
        clips = self.synthesize(ctrl, g_cond)

        self.log_jerk(clips[:,:,:self.pose_dim], log_prefix)
        file_name = f"{self.current_epoch}_{self.global_step}_{log_prefix}"
        self.log_results(clips.cpu().detach().numpy(), file_name, log_prefix, render_video=False)

    def test_step(self, batch, batch_idx):
        loss = self(batch)

        self.synthesize_and_log(batch, "test")

        output = {"test_loss": loss}
        return output
        
    def synthesize(self, ctrl, global_cond):
        print("synthesize")

        training_noise_schedule = self.noise_schedule.to(ctrl.device)
        inference_noise_schedule = training_noise_schedule

        talpha = 1 - training_noise_schedule
        talpha_cum = torch.cumprod(talpha, dim=0)

        beta = inference_noise_schedule
        alpha = 1 - beta
        alpha_cum = torch.cumprod(alpha, dim=0)

        T = []
        for s in range(len(inference_noise_schedule)):
            for t in range(len(training_noise_schedule) - 1):
                if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
                    twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
                    T.append(t + twiddle)
                    break
                    

        if len(ctrl.shape) == 2:# Expand rank 2 tensors by adding a batch dimension.
            ctrl = ctrl.unsqueeze(0)
            global_cond = global_cond.unsqueeze(0)
        poses = torch.randn(ctrl.shape[0], ctrl.shape[1], self.pose_dim, device=ctrl.device)
            
        nbatch = poses.size(0)
        noise_scale = (alpha_cum**0.5).type_as(poses).unsqueeze(1)

        for n in range(len(alpha) - 1, -1, -1):
            c1 = 1 / alpha[n]**0.5
            c2 = beta[n] / (1 - alpha_cum[n])**0.5
                                        
            poses = c1 * (poses - c2 * self.diffusion_model(poses, ctrl, global_cond, T[n].unsqueeze(-1)).squeeze(1))

            if n > 0:
                noise = torch.randn_like(poses)
                sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
                poses += sigma * noise
                
        anim_clip = self.destandardizeOutput(poses)
        if not self.unconditional:
            out_ctrl = self.destandardizeInput(ctrl)
            anim_clip = torch.cat((anim_clip, out_ctrl), dim=2) 

        return anim_clip
    

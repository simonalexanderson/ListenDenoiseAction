# Copyright 2023 Motorica AB, Inc. All Rights Reserved.

import io
import json
from pathlib import Path
import joblib as jl
import numpy as np
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.pipeline import Pipeline
from pymo.writers import *
from pymo.data import Joint, MocapData
from pymo.preprocessing import *
from pymo.viz_tools import *
from scipy import interpolate


class LoggingMixin:

    def log_results(self, pred_clips, file_name, log_prefix, logdir=None, render_video=True):

        if logdir is None:
            logdir = f"{self.logger.save_dir}/{self.logger.name}/version_{self.logger.version}/"

        if len(log_prefix.strip())>0:
            file_name = file_name + "_" + log_prefix
            
        bvh_data = self.feats_to_bvh(pred_clips)
        nclips = len(bvh_data)
        framerate = np.rint(1/bvh_data[0].framerate)
        
        if self.hparams.Validation["max_render_clips"]:
            nclips = min(nclips, self.hparams.Validation["max_render_clips"])
        
        self.write_bvh(bvh_data[:nclips], log_dir=logdir, name_prefix=file_name)
        
        if render_video:
            pos_data = self.bvh_to_pos(bvh_data)
            
        if render_video:
            self.render_video(pos_data[:nclips], log_dir=logdir, name_prefix=file_name)
                    
        
    def feats_to_bvh(self, pred_clips):
        #import pdb;pdb.set_trace()
        data_pipeline = jl.load(Path(self.hparams.dataset_root) / self.hparams.Data["datapipe_filename"])
        n_feats = data_pipeline["cnt"].n_features        
        data_pipeline["root"].separate_root=False
        
        print('inverse_transform...')
        bvh_data=data_pipeline.inverse_transform(pred_clips[:,:,:n_feats])
        return bvh_data
        
    def write_bvh(self, bvh_data, log_dir="", name_prefix=""):
        writer = BVHWriter()
        nclips = len(bvh_data)
        for i in range(nclips):        
            if nclips>1:
                fname = f"{log_dir}/{name_prefix}_{str(i).zfill(3)}.bvh"
            else:
                fname = f"{log_dir}/{name_prefix}.bvh"
            print('writing:' + fname)
            with open(fname,'w') as f:
                writer.write(bvh_data[i], f)
        
    def bvh_to_pos(self, bvh_data):        
        # convert to joint positions
        return MocapParameterizer('position').fit_transform(bvh_data)
                
    def render_video(self, pos_data, log_dir="", name_prefix=""):
        # write bvh and skeleton motion
        nclips = len(pos_data)
        for i in range(nclips):        
            if nclips>1:
                fname = f"{log_dir}/{name_prefix}_{str(i).zfill(3)}"
            else:
                fname = f"{log_dir}/{name_prefix}"
            print('writing:' + fname + ".mp4")
            render_mp4(pos_data[i], fname + ".mp4", axis_scale=200)
        
            
    def log_jerk(self, x, log_prefix):

        deriv = x[:, 1:] - x[:, :-1]
        acc = deriv[:, 1:] - deriv[:, :-1]
        jerk = acc[:, 1:] - acc[:, :-1]
        self.log(f'{log_prefix}_jerk', torch.mean(torch.abs(jerk)), sync_dist=True)
        

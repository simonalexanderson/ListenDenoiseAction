# Copyright 2023 Motorica AB, Inc. All Rights Reserved.

import os
import random
import argparse
import json
import torch
import torch.utils.data
import sys
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle as pkl
import pandas as pd
from scipy import interpolate

def concat_dataframes(x,y):
    # Assume data is synched on the start time
    if x.shape[0]<y.shape[0]:
        y=y[:x.shape[0]]
        y.index=x.index
        return pd.merge_asof(x, y, on='time', tolerance=pd.Timedelta('0.01s')).set_index('time')
    else:
        x=x[:y.shape[0]]
        y.index=x.index
        return pd.merge_asof(x, y, on='time', tolerance=pd.Timedelta('0.01s')).set_index('time')
        
def nans2zeros(x):
    ii = np.where(np.isinf(x))
    x[ii]=0
    ii = np.where(np.isnan(x))
    x[ii]=0
    return x
        
def dataframe_nansinf2zeros(df):
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    return df
    
def align_start(x,y):
    if x.shape[0]<y.shape[0]:
        return x,y[:x.shape[0]]
    else:
        return x[:y.shape[0],:],y
        
def parse_token(f_name, inds):
    basename = os.path.basename(f_name).split('.')[0]
    tokens = basename.split('_')
    out=""
    assert(len(inds)>0)
    for i in range(len(inds)):
        assert len(tokens) > inds[i], f"{inds[i]} out of range in {basename}"
        out+=tokens[inds[i]]
        if i<len(inds)-1:
          out+="_"
    return out
    
def styles2onehot(all_styles, style_token):
    oh = np.zeros((len(all_styles)))
    for i in range(len(all_styles)):
        if style_token == all_styles[i]:
            oh[i] = 1
            return oh

    print("Style token error. Not found " + style_token)    
    
def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files

def resample_data(data, nframes_new, has_root_motion, mode='linear'):

    nframes = data.shape[0]
    x = np.arange(0, nframes)/(nframes-1)
    xnew = np.arange(0, nframes_new)/(nframes_new-1)

    data_out = np.zeros((nframes_new, data.shape[1]))
    for jj in range(data.shape[1]):
        y = data[:,jj]
        f = interpolate.interp1d(x, y, bounds_error=False, kind=mode, fill_value='extrapolate')
        data_out[:,jj] = f(xnew)
    
    #Scale root deltas to match new frame-rate
    if has_root_motion:
        sc = nframes/nframes_new
        data_out[:,-3:] = data_out[:,-3:]*sc
    return torch.from_numpy(data_out).float()
        
class MotionDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, datafiles_file, data_hparams=None):
        files = files_to_list(datafiles_file)
        self.timestretch_prob = data_hparams["timestretch_prob"] if "timestretch_prob" in data_hparams else 0.1
        self.timestretch_factor = data_hparams["timestretch_factor"] if "timestretch_factor" in data_hparams else 0
        self.segment_length = data_hparams["segment_length"]

        # when augmenting with timestretched data, we cut into longer sequences
        max_segment_length = int(self.segment_length*(1.0 + self.timestretch_factor))
                    
        start_idx=0
        data = {"input":[], "output":[], "styles":[]}
        indexes = []

        for fi in range(len(files)):
            fname = files[fi]
            if fname == "":
                print("No file at line: " + str(fi))
                continue

            #input conditioning
            in_mod = data_hparams["input_modality"]
            indata_file = Path(data_root) / f'{fname}.{in_mod}.pkl'
            infeats_file = Path(data_root) / data_hparams["input_feats_file"]
            infeats_cols = np.loadtxt(infeats_file, dtype=str).tolist()
            with open(indata_file, 'rb') as f:
                in_feats = dataframe_nansinf2zeros(pkl.load(f).astype('float32'))
                in_feats = in_feats[infeats_cols].values
                self.n_input = in_feats.shape[1]
                
            #output
            out_mod = data_hparams["output_modality"]
            outdata_file = Path(data_root) / f'{fname}.{out_mod}.pkl'
            with open(outdata_file, 'rb') as f:
                out_feats = dataframe_nansinf2zeros(pkl.load(f).astype('float32')).values
                self.n_output = out_feats.shape[1]
                
            in_feats, out_feats = align_start(in_feats, out_feats)

            # Optionally trim edges (may contain TPose etc)
            trim_edges = data_hparams["trim_edges"]
            if trim_edges>0:
                in_feats = in_feats[trim_edges:-trim_edges]
                out_feats = out_feats[trim_edges:-trim_edges]
                
            n_frames=in_feats.shape[0]

            # global conditioning (from file naming convention)
            if "styles_file" in data_hparams:
                styles_file = Path(data_root) / data_hparams["styles_file"]
                all_styles = np.loadtxt(styles_file, dtype=str).tolist()            
                styles_oh = np.tile(styles2onehot(all_styles, parse_token(files[fi], data_hparams["style_index"])),(n_frames,1))
                self.n_styles = len(all_styles)
            else:
                self.n_styles = 0

            #we create indexes for full length sequences here
            seglen=max_segment_length
            if n_frames >= seglen:
                idx_array = torch.arange(start_idx, start_idx + n_frames).unfold(
                        0, seglen, 1
                    )
                data["input"].append(in_feats)
                data["output"].append(out_feats)
                if self.n_styles>0:
                    data["styles"].append(styles_oh)
                indexes.append(idx_array)                
                start_idx += n_frames
                
        #flatten vertically and make into a torch tensor
        data["input"]=torch.from_numpy(np.vstack(data["input"])).float()
        data["output"]=torch.from_numpy(np.vstack(data["output"])).float()
        if self.n_styles>0:
            data["styles"]=torch.from_numpy(np.vstack(data["styles"])).float()        
        print(f"=== tot number of frames: {data['output'].shape[0]} =====")

        self.data = data

        indexes=torch.cat(indexes, dim=0)
        self.indexes = indexes[torch.randperm(indexes.size(0))]
        
    def assert_not_const(self, data):
        eps = 1e-6
        assert((data.std(axis=0)<eps).sum()==0)
    
    def fit_scalers(self):

        in_scaler = StandardScaler()
        self.assert_not_const(self.data["input"])
        in_scaler.fit(self.data["input"])
            
        self.assert_not_const(self.data["output"])
        out_scaler = StandardScaler()
        out_scaler.fit(self.data["output"])
        
        if self.n_styles>0:
            style_scaler = StandardScaler()
            style_scaler.mean_=np.zeros(self.n_styles)
            style_scaler.scale_=np.ones(self.n_styles)
        else:
            style_scaler=None

        return {"in_scaler": in_scaler, "style_scaler": style_scaler,"out_scaler": out_scaler}

    def standardize(self, scalers):   
        self.data["input"] = torch.from_numpy(scalers["in_scaler"].transform(self.data["input"])).float()
        self.data["output"] = torch.from_numpy(scalers["out_scaler"].transform(self.data["output"])).float()
        
    def timestretch(self, data, segment_length, factor, has_root_motion=False):
        if factor<1.0:
            #Truncate original samples and stretch
            return resample_data(data[:int(factor*segment_length)],segment_length, has_root_motion)
        elif factor>1.0:
            #Stretch original samples and trunkate
            return resample_data(data,int(factor*segment_length),has_root_motion)[:segment_length]
        else:
            # return original
            return data[:segment_length]
        
    def __getitem__(self, index):
        in_feats = self.data["input"][self.indexes[index]]
        out_feats = self.data["output"][self.indexes[index]]
        
        if self.timestretch_factor>0:
            #note that the sequences are longer than specified so we can resample faster speeds
            if torch.rand((1,))<self.timestretch_prob:            
                #resample and cut to specified seq len
                segment_length = self.segment_length
                factor = torch.rand((1,))*self.timestretch_factor*2-self.timestretch_factor + 1
                in_feats = self.timestretch(in_feats, segment_length, factor, has_root_motion=False)
                out_feats = self.timestretch(out_feats, segment_length, factor, has_root_motion=True)
            else:
                #just cut to specified seq len
                in_feats = in_feats[:self.segment_length]
                out_feats = out_feats[:self.segment_length]
        
        if self.n_styles>0:
            styles = self.data["styles"][self.indexes[index]]
            styles = styles[:self.segment_length]
        else:
            styles = []

        return (in_feats, styles, out_feats)

    def __len__(self):
        return self.indexes.size(0)
              
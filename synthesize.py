# Copyright 2023 Motorica AB, Inc. All Rights Reserved.
from os.path import join
import os, sys, getopt
import torch
import numpy as np
import pickle as pkl
from pytorch_lightning import Trainer, seed_everything
from utils.motion_dataset import styles2onehot, nans2zeros
from models.LightningModel import LitLDA

def sample_mixmodels(models, batches, guidance_factors):
    # asserts that the models are compatible, 
    # i.e. they have the same number of noise steps, pose dim and pose scalers
    assert len(guidance_factors)==(len(models)-1), "n_guidance_factors should be eq to n_models-1"
    noise_sched_0 = models[0].noise_schedule
    o_scaler_0 = models[0].hparams["Data"]["scalers"]["out_scaler"]
    eps = 0.000001
    for i in range(1, len(models)):
        # models should have same noise schedule        
        assert torch.all(torch.abs(models[i].noise_schedule - noise_sched_0)<eps), "different noise-schedule"
        
        # models should have same out scalers
        o_scaler_i = models[i].hparams["Data"]["scalers"]["out_scaler"]
        assert np.all(np.abs(o_scaler_i.mean_-o_scaler_0.mean_)<eps), "different pose standardization"
        assert np.all(np.abs(o_scaler_i.scale_-o_scaler_0.scale_)<eps), "different pose standardization"
                
    beta = np.array(noise_sched_0)

    talpha = 1 - beta
    talpha_cum = np.cumprod(talpha)

    alpha = 1 - beta
    alpha_cum = np.cumprod(alpha)
    T = np.arange(0,len(beta), dtype=np.float32)

    ctrl, global_cond, _ = batches[-1]
    poses = torch.randn(ctrl.shape[0], ctrl.shape[1], models[0].pose_dim, device=models[0].device)
               
    nbatch = poses.size(0)
    noise_scale = torch.from_numpy(alpha_cum**0.5).type_as(poses).unsqueeze(1)

    for n in range(len(alpha) - 1, -1, -1):
        c1 = 1 / alpha[n]**0.5
        c2 = beta[n] / (1 - alpha_cum[n])**0.5
                                    
        diffs = []
        for i, model in enumerate(models):
            l_cond, g_cond, _ = batches[i]
            diffs.append(model.diffusion_model(poses, l_cond, g_cond, torch.tensor([T[n]], device=poses.device)).squeeze(1))
            
        diff0=diffs[0]
        diff=diff0
        for i in range(len(guidance_factors)):
            diff += guidance_factors[i]*(diffs[i+1] - diff0)         
        
        poses = c1 * (poses - c2 * diff)
            
        if n > 0:
            noise = torch.randn_like(poses)
            sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
            poses += sigma * noise
             
    out_poses = models[-1].destandardizeOutput(poses)
    if not models[-1].unconditional:
        out_ctrl = models[-1].destandardizeInput(ctrl)
        anim_clip = torch.cat((out_poses, out_ctrl), dim=2).cpu().detach().numpy()         
    else:
        anim_clip = out_poses.cpu().detach().numpy()
    return anim_clip

def do_synthesize(models, l_conds, g_conds, file_name, postfix, trim, dest_dir, guidance_factors, gpu, render_video, outfile):
    nframes = l_conds[-1].size(1)
    
    device = torch.device(gpu)
    batches = []
    for i in range(len(models)):
        models[i].to(device)
        models[i].eval()
        batch = l_conds[i].to(device) if len(l_conds[i])>0 else [], g_conds[i].to(device) if len(g_conds[i])>0 else [], None    
        batches.append(batch)
    
    with torch.no_grad():
        clips = sample_mixmodels(models, batches, guidance_factors)        
        models[-1].log_results(clips[:,trim:nframes-trim,:], outfile, "", logdir=dest_dir, render_video=render_video)

def nans2zeros(x):
    ii = np.where(np.isinf(x))
    x[ii]=0
    ii = np.where(np.isnan(x))
    x[ii]=0
    return x
    
def get_style_vector(styles_file, style_token, nbatch, nframes):
    all_styles = np.loadtxt(styles_file, dtype=str)    
    styles_onehot = styles2onehot(all_styles, style_token)
    styles = styles_onehot.repeat(nbatch, nframes,1)    

def get_cond(model, data_dir, input_file, style_token, length):
    # Load input features
    with open(join(data_dir, input_file), 'rb') as f:
        ctrl = pkl.load(f)
    ctrl = ctrl[startframe:]
    if endframe>0 and endframe<ctrl.shape[0]:
        ctrl = ctrl[:endframe]
    input_feats_file = os.path.join(data_dir, model.hparams.Data["input_feats_file"])
    input_feats = np.loadtxt(input_feats_file, dtype=str)
    ctrl = ctrl[input_feats]
    ctrl = nans2zeros(torch.from_numpy(ctrl.values).float().unsqueeze(0))
    
    nbatch = ctrl.size(0)
    nframes = ctrl.size(1)
    
    # parse styles
    styles=[]
    if "styles_file" in model.hparams.Data:   
        styles_file = os.path.join(data_dir, model.hparams.Data["styles_file"])
        all_styles = np.loadtxt(styles_file, dtype=str)
        
        styles_onehot = torch.from_numpy(styles2onehot(all_styles, style_token)).float()
        styles = styles_onehot.repeat(nbatch, nframes,1)    
        
    return model.standardizeInput(ctrl), styles

def arg2tokens(arg, delim=","):
    return arg.strip().split(delim)
    
def arg2tokens_f(arg, delim=","):
    ts=arg2tokens(arg, delim)
    out=[]
    for t in ts:
        out.append(float(t))
    return out

if __name__ == "__main__":
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv,"hc:x:d:f:s:e:t:r:p:g:k:v:o",["checkpoints=", "data_dirs=", "input_files=", "styles=", "start=", "end=", "trim=", "seed=", "postfix=", "dest_dir=", "gf=", "gpu=", "video=", "outfile="])
    except getopt.GetoptError:
        print ('python synthesize.py -c checkpoint -d data_dir -i input_file -s style -b start -e end -r seed -p postfix -l dest_dir -g gf -k gpu -v video -o outfile')
        
        sys.exit(2)

    trim = 0
    postfix=""
    dest_dir="results"
    seed=42
    startframe=0    
    guidance_factors = []
    gpu="cuda:0"
    style_tokens=None
    render_video=True
    outfile=""

    for opt, arg in opts:
        if opt == '-h':
            print ('python synthesize.py -c checkpoint -d data_dir -i input_file -s style -b start -e end')
            print ('example usage: python synthesize.py --checkpoint=results/moglow/styleloco/lightning_logs/version_9/checkpoints/epoch\=8-step\=146105.ckpt --data_dir=data/motorica/locomotion/processed_sm6_6/ --input_file=data/motorica/locomotion/processed_sm6_6/loco_act01_male_w65_h178_earth_ex05_mix_q03_2022-02-02_001.expmap_20fps.pkl --style=act01_earth --end=200 --model=moglow --seed=seed')
            sys.exit()
        elif opt in ("-c", "--checkpoints"):
            checkpoints = arg2tokens(arg)
        elif opt in ("-d", "--data_dirs"):
            data_dirs = arg2tokens(arg)
        elif opt in ("-f", "--input_files"):
            input_files = arg2tokens(arg)
        elif opt in ("-s", "--styles"):
            style_tokens = arg2tokens(arg)
        elif opt in ("-b", "--start"):
            startframe = int(arg)
        elif opt in ("-e", "--end"):
            endframe = int(arg)
        elif opt in ("-g", "--gf"):
            guidance_factors = arg2tokens_f(arg)
        elif opt in ("-t", "--trim"):
            trim = int(arg)
        elif opt in ("-r", "--seed"):
            seed = int(arg)
        elif opt in ("-p", "--postfix"):
            postfix = arg
        elif opt in ("-l", "--dest_dir"):
            dest_dir = arg
        elif opt in ("-k", "--gpu"):
            gpu = arg
        elif opt in ("-v", "--video"):
            render_video = arg.lower()=="true" 
        elif opt in ("-o", "--outfile"):
            outfile = arg 

    out_file_name = os.path.basename(input_files[0]).split('.')[0]
    seed_everything(seed)
    models = []
    l_conds = []
    g_conds = []
    for i in range(len(checkpoints)):
        model = LitLDA.load_from_checkpoint(checkpoints[i],dataset_root=data_dirs[i])
        models.append(model)
        if style_tokens is not None:
            l_cond, style = get_cond(model, data_dirs[i], input_files[i], style_tokens[i], endframe)
        else:
            l_cond, style = get_cond(model, data_dirs[i], input_files[i], "", endframe)
        l_conds.append(l_cond)
        g_conds.append(style)
    
    do_synthesize(models, l_conds, g_conds, out_file_name, postfix, trim, dest_dir, guidance_factors, gpu, render_video, outfile)

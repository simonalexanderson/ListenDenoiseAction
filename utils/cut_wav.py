# Copyright 2023 Motorica AB, Inc. All Rights Reserved.

import numpy as np
import scipy.io.wavfile as wav

import os
import sys
    
if __name__ == "__main__":        
    if len(sys.argv)==6:
        filename = sys.argv[1]
        starttime = float(sys.argv[2])
        endtime = float(sys.argv[3])
        suffix = sys.argv[4]
        dest_dir = sys.argv[5]
    else:
        print("usage: python cut_wav.py starttime(s) endtime(s) suffix dest_dir")
        sys.exit(0)
        
    #import pdb;pdb.set_trace()
    print(f'Cutting AUDIO {filename} from: {starttime} to {endtime}')
    basename = os.path.splitext(os.path.basename(filename))[0]
    outfile = os.path.join(dest_dir, basename+"_"+suffix+'.wav')
    fs,X = wav.read(filename)
    start_idx = int(np.round(starttime*fs))
    end_idx = int(np.round(endtime*fs))
    if end_idx<X.shape[0]:
        wav.write(outfile, fs, X[start_idx:end_idx])
    else:
        print("EOF REACHED")

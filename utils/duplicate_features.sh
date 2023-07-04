#!/bin/bash
#./duplicate_features.sh data/GENEA/processed_60fps audio14_60fps
augmentation=mirrored
find $1 -name "*.${2}.pkl" -not -name "*_${augmentation}*.${2}.pkl" -print0 | xargs -0 -I {} basename -z {} .${2}.pkl | xargs -0 -I {} cp $1/{}.${2}.pkl $1/{}_${augmentation}.${2}.pkl
#find $1 -name "*.${2}.npy" -print0 | xargs -0 -I {} basename -z {} .${2}.npy | xargs -0 -I {} echo {}_${augmentation}.${2}.npy
#find $1 -name "*.${2}.npy" -print0 | xargs -0 -I {} basename {} .${2}.npy

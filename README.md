# Listen Denoise Action
This repository contains code for reproducing the paper [Listen, denoise, action!
Audio-driven motion synthesis with diffusion models](https://arxiv.org/abs/2211.09707).

Please watch the following video for an introduction to the paper:
* SIGGRAPH 2023 Presentation: [https://youtu.be/Qfd2EpzWgok](https://youtu.be/Qfd2EpzWgok)

For samples and overview, please see our [project page](https://www.speech.kth.se/research/listen-denoise-action/).

## Installation
We provide a Docker file and requirements.txt for installation using a docker image or conda.

### Installation using conda
```
conda install python=3.9
conda install -c conda-forge mpi4py mpic
pip install -r requirements.txt
```

## Data and pretrained models
Download the data and pretrained models from [here]() and move the data folders to 'data' and the checkpoints to 'pretrained_models'.

## Model Training
```
python train.py <data_dir> <hparams_file>
```

Example:
```
python train.py data/motorica_dance/ hparams/dance_LDA.yaml
```
## Synthesis
We provide shell scripts for reproducing the user studies in the paper. To try out locomtion synthesis, please go to http://motorica.ai
```
./experiments/dance_LDA.sh
./experiments/dance_LDA-U.sh
./experiments/tsg_LDA.sh
./experiments/zeggs_LDA.sh
./experiments/zeggs_LDA-G.sh
```

## Citation
```
@article{alexanderson2023listen,
    title={Listen, Denoise, Action! Audio-Driven Motion Synthesis with Diffusion Models},
    author={Alexanderson, Simon and Nagy, Rajmund and Beskow, Jonas and Henter, Gustav Eje},
    journal={ACM Trans. Graph.},
    volume={42},
    number={4},
    pages={1--20},
    doi={10.1145/3592458},
    year={2023}
}
```


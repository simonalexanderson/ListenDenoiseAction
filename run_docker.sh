docker run -it --rm --gpus '"device=0,1,2,3,4,5,6,7"' -v $PWD:/workspace/dockers --ipc=host -v=$HOME/data:/workspace/dockers/data simonal_diffusion /bin/sh -c 'cd dockers; bash'

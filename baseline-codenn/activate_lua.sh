#!/bin/bash

###
### Activate lua 5.2, make sure we are using the correct version
###
torchactivate=''
if [[ $(hostname -s) = ash  && -f /scratch/software/torch/install/bin/torch-activate ]]; then
    printf "ash: activate lua 5.2 ...\n"
    torchactivate=/scratch/software/torch/install/bin/torch-activate
elif [[ $(hostname -s) = bishop && -f /home/siyuan/torch/install/bin/torch-activate ]]; then
    printf "bishop: activate lua 5.2 ...\n"
    torchactivate=/home/siyuan/torch/install/bin/torch-activate
elif [ -f /scratch/software/torch/install/bin/torch-activate ]; then
    torchactivate=/scratch/software/torch/install/bin/torch-activate
fi

if [[ -z $torchactivate ]]; then
    printf "\n!!!Cannot find torch-activate. \n!!!Continue... \n!!!Make sure you have activated the lua 5.2 instead LuaJIT.\n\n"
else
    . $torchactivate
fi

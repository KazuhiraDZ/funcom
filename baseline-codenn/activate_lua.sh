#!/bin/bash

###
### Activate lua 5.2, make sure we are using the correct version
###
if [[ $(hostname -s) = ash ]]; then
    printf "ash: activate lua 5.2 ...\n"
    if [ -f /scratch/software/torch/install/bin/torch-activate ]; then
	. /scratch/software/torch/install/bin/torch-activate
    else
	echo "Cannot find torch in /scratch/software/torch/install/bin/torch-activate. Exit."
	exit 1
    fi
else
    if [ -f /scratch/software/torch/install/bin/torch-activate ]; then
	. /scratch/software/torch/install/bin/torch-activate
    else
	echo "Cannot find torch in /scratch/software/torch/install/bin/torch-activate."
	printf "\n***\n***make sure you are using lua 5.2\n***\n"    
    fi
fi

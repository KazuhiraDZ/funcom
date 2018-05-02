#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

# path to moses decoder: https://github.com/moses-smt/mosesdecoder

if [[ "$(hostname -s)" == "ash" ]]; then
    export mosesdecoder=/scratch/commitgen/mosesdecoder
elif [[ "$(hostname -s)" == "newt" ]]; then
    export mosesdecoder=/scratch/commitgen/moses/mosesdecoder
else
    export mosesdecoder=''
    echo "\nDoes not recognize the server.\n"
fi

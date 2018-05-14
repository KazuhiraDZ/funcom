#!/bin/bash

show_usage() {
    echo -e "Usage: $0 reference_file prediction_file"
}

if [ $# -lt 2 ]
then
    show_usage
    exit 1
fi

ref=$1
pred=$2

if [ ! -f 'multi-bleu.perl' ]; then
    curl -O "https://raw.githubusercontent.com/moses-smt/mosesdecoder/RELEASE-4.0/scripts/generic/multi-bleu.perl"
    echo "downloaded multi-bleu.perl from moses"
fi

perl ./multi-bleu.perl $ref < $pred

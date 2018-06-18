#!/bin/bash

function checkfiles ()
{
    files=("$@")
    for file in "${files[@]}";
    do
	if [ ! -s "$file" ]
	then
    	    echo "train.sh: $file does not exist. Exit."
    	    exit 0
	fi
    done
}

if [ "$#" -ne 5 ] && [ "$#" -ne 6 ]; then
    echo "train.sh: Illegal number of parameters"
    echo "train.sh: Usage: $0 output_directory_for_models data_directory_for_training vocab_size_for_source vocab_size_for_target [optional valid freq: default 10k]"
    exit 0
fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
source ~/.profile

nematus=./nematus-tensorflow/nematus/nmt.py

modelout=$1
datadir=$2
vocabsize_src=$3
vocabsize_tgt=$4
maxlen=$5
validfreq=${6:-10000} # 10000

if test "$(ls -A "$modelout")"; then
    echo "train.sh: the output directory for model files is not empty."
    exit 0
else
    mkdir -p $modelout
fi

trainsrc=$datadir/train.src.txt
traintgt=$datadir/train.tgt.txt
validsrc=$datadir/valid.src.txt
validtgt=$datadir/valid.tgt.txt
vocabsrc=$datadir/vocab.src.json
vocabtgt=$datadir/vocab.tgt.json

files=("$trainsrc" "$traintgt" "$validsrc" "$validtgt" "$vocabsrc" "$vocabtgt")
echo "train.sh: check files..."
checkfiles "${files[@]}"

echo "train.sh: dictionaries $datadir/vocab.src.json $datadir/vocab.tgt.json"
echo "train.sh: valid freq: $validfreq"

python2.7 $nematus \
	  --model $modelout/model.npz \
	  --dim_word 512 \
	  --dim 1024 \
	  --source_vocab_size $vocabsize_src \
	  --target_vocab_size $vocabsize_tgt \
	  --decay_c 0 \
	  --clip_c 1 \
	  --lrate 0.0001 \
	  --optimizer adam \
	  --maxlen $maxlen \
	  --batch_size 80 \
	  --valid_batch_size 80 \
	  --datasets $trainsrc $traintgt \
	  --valid_source_dataset $validsrc \
	  --valid_target_dataset $validtgt \
	  --dictionaries $vocabsrc $vocabtgt \
	  --validFreq $validfreq \
	  --dispFreq 1000 \
	  --dropout_embedding 0.2 \
	  --dropout_hidden 0.2 \
	  --dropout_source 0.1 \
	  --dropout_target 0.1 \
	  --no_shuffle \
	  --saveFreq 30000 \
	  --sampleFreq 10000

#!/bin/sh

# downloaded from https://raw.githubusercontent.com/rsennrich/wmt16-scripts/master/sample/translate.sh
# command time bash test.sh workdir/models/model.npz-30000 workdir/test.src.txt workdir/predict.txt workdir/models/model.npz.json

nematus=./nematus-tensorflow/nematus

if [ "$#" -ne 4 ]; then
    echo "usage: model testfile testoutput npz.json_file"
    exit 1
fi

MODEL=$1 #./models/model.100k.npz.npz.best_bleu
TEST=$2  #./data/test.1k.diff
OUT=$3   #./data/test.1k.diff.output
JSONORIG=$4

printf "model=\"%s\", test=%s, out=%s, jsonorig=%s\n" "$MODEL" "$TEST" "$OUT" "$JSONORIG"

models=($MODEL)
for modelfile in "${models[@]}"
do
    # example: models/model.205552.50000.50000.npz.json models/model.205552.50000.50000.iter120000.npz.json
    jsonfile="${modelfile}.json"
    if [ ! -f $jsonfile ]; then
	cp $JSONORIG $jsonfile
    fi
done

python2.7 $nematus/translate.py \
	    -m $MODEL \
	    -i $TEST \
	    -o $OUT \
	    -k 12 -n -p 1


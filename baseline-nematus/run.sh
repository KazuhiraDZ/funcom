#!/bin/bash
source time.sh

today=`date +%Y-%m-%d.%H%M%S`
defaultlog=run.log.$today

if [ "$#" -ne 1 ] && [ "$#" -ne 2 ]; then
    echo "$0: Illegal number of parameters"
    echo "$0: Usage: $0 [config_file] [optional: logfile; default: $defaultlog]"
    exit 0
fi

config=$1
log=${2:-$defaultlog}
echo "config file: $config, log file: $log"

tensorflowurl=https://github.com/EdinburghNLP/nematus/archive/tensorflow.zip

function downloadnematus()
{
    echo "git submodule does not work... it's okay..."

    local nematusdir=nematus-tensorflow
    if [ -d "$nematusdir" ]; then
	echo "nematus-tensorflow already exists. if you would like to update the nematus copy, remove the folder first."
    else
	echo "we download from my fork..."
	# use my fork of nematus, in case there is any major change in the original nematus repo
	git clone --depth=1 --branch=tensorflow https://github.com/sjiang1/nematus.git $nematusdir
	rm -rf ${nematusdir}/.git	
    fi
}


git submodule init
if [ $? -eq 0 ]
then
    git submodule update
    if [ $? -ne 0 ]
    then
	downloadnematus
    fi
else
    downloadnematus
fi


start=$(date +%s.%N)
python3 prepdata.py --config $config 2>&1 | tee -a $log
end=$(date +%s.%N)
diff=`show_time $end $start`
echo "prepdata: $diff" | tee -a $log

function checkconfig()
{
    local var=$1
    if [ -z "${TRAIN[$var]}" ]; then
        echo "$0: cannot get config variable: $var for train.sh. Exit."
        exit 1
    fi
    printf "$var: ${TRAIN[$var]}, "
}

eval "$(cat $config  | python ./ini2arr.py)"
printf "train.sh: "
checkconfig 'outdir'
checkconfig 'data'
checkconfig 'vocabsize_src'
checkconfig 'vocabsize_tgt'
printf "\n"
start=$(date +%s.%N)
bash train.sh ${TRAIN[outdir]} ${TRAIN[data]} ${TRAIN[vocabsize_src]} ${TRAIN[vocabsize_tgt]} | tee -a $log
end=$(date +%s.%N)
diff=`show_time $end $start`
echo "prepdata: $diff" | tee -a $log

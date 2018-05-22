#!/bin/bash
source time.sh

today=`date +%Y-%m-%d.%H%M%S`
log=run.log.$today

read -r -d '' helpmsg <<EOM
Usage: $0
    -c          [required] set a config file
    -d          [optional] set a gpu id to be used
    -l          [optional] set the file name for log; default file name: $log
    -h          display this help message
EOM

passarg=true
while getopts ":c:d:l:h" opt; do
  case ${opt} in
    c )
	config=$OPTARG
	;;
    d )
	dev=$OPTARG
	;;
    l )
	log=$OPTARG
	;;
    h )
        passarg=false
	;;
    \? )
	echo "Invalid option: $OPTARG" 1>&2
	passarg=false
	;;
    : )
	echo "Invalid option: $OPTARG requires an argument" 1>&2
	passarg=false
	;;
  esac
done
shift $((OPTIND -1))

if $passarg && [ -z "$config" ]; then
    echo "Must use -c to specify a config file to use."
    passarg=false
fi

if ! $passarg ;then
    echo "$helpmsg"
fi

echo "config file: $config, log file: $log" | tee -a $log


###
### download nematus if necessary
###
function downloadnematus()
{
    echo "git submodule does not work... it's okay..."

    local nematusdir=nematus-tensorflow
    if [ -d "$nematusdir" ]; then
	echo "nematus-tensorflow already exists. if you would like to update the nematus copy, remove the folder first." | tee -a $log
    else
	echo "we download from my fork..." | tee -a $log
	# use my fork of nematus, in case there is any major change in the original nematus repo
	git clone --depth=1 --branch=tensorflow https://github.com/sjiang1/nematus.git $nematusdir
	rm -rf ${nematusdir}/.git
    fi
}

git submodule init
submoduleflag=false
if [ $? -eq 0 ]
then
    git submodule update
    if [ $? -eq 0 ]
    then
	submoduleflag=true
    fi
fi

if ! $submoduleflag ;then
    downloadnematus
fi

###
### prepare the data set
###
start=$(date +%s.%N)
python3 prepdata.py --config $config 2>&1 | tee -a $log
end=$(date +%s.%N)
diff=`show_time $end $start`
echo "prepdata: $diff" | tee -a $log

###
### train nematus
###
function checkconfig()
{
    local var=$1
    if [ -z "${TRAIN[$var]}" ]; then
        echo "$0: cannot get config variable: $var for train_nmt.sh. Exit."
        exit 1
    fi
    printf "$var: ${TRAIN[$var]}, " | tee -a $log
}

eval "$(cat $config  | python ./ini2arr.py)"
printf "train_nmt.sh: " | tee -a $log
checkconfig 'outdir'
checkconfig 'data'
checkconfig 'vocabsize_src'
checkconfig 'vocabsize_tgt'
printf "\n" | tee -a $log
start=$(date +%s.%N)
if [ -z "$dev" ]; then
    bash train_nmt.sh ${TRAIN[outdir]} ${TRAIN[data]} ${TRAIN[vocabsize_src]} ${TRAIN[vocabsize_tgt]} 2>&1 | tee -a $log
else
    CUDA_VISIBLE_DEVICES=$dev bash train_nmt.sh ${TRAIN[outdir]} ${TRAIN[data]} ${TRAIN[vocabsize_src]} ${TRAIN[vocabsize_tgt]} 2>&1 | tee -a $log
fi

end=$(date +%s.%N)
diff=`show_time $end $start`
echo "train: $diff" | tee -a $log

exit 0

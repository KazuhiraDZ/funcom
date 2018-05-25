#!/bin/bash
source time.sh

today=`date +%Y-%m-%d.%H%M%S`
log=train.log.$today

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
    exit 0
fi

echo "config file: $config, log file: $log" | tee -a $log
exec {BASH_XTRACEFD}>>$log
set -x

if [[ $(hostname -s) = ash ]]; then
    printf "ash: source sourceme.sh ...\n"
    if [ -f /scratch/funcom/sourceme.sh ]; then
	source /scratch/funcom/sourceme.sh
    else
	echo "Cannot find /scratch/funcom/sourceme.sh. Exit."
	exit 1
    fi
else
    printf "\n***\n***make sure you source sourcme.sh from Alex\n***\n"
fi

source download_nematus.sh

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

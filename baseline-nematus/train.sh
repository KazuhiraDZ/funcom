#!/bin/bash
source time.sh

today=`date +%Y-%m-%d.%H%M%S`
log=train.log.$today

RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

function infoecho(){ printf "$1" | tee -a $log; }
function warning(){ echo -e "${YELLOW}Warning: $1${NC}" | tee -a $log; }
function error(){ echo -e "${RED}Error: $1${NC}" | tee -a $log; }

read -r -d '' helpmsg <<EOM
Usage: $0
    -c          [required] set a config file
    -d          [required] set a gpu id to be used
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

if $passarg && [ -z "$dev" ]; then
    echo "Must use -d to specify a gpu to use."
    passarg=false
fi

if ! $passarg ;then
    echo "$helpmsg"
    exit 0
fi

infoecho "config file: $config, log file: $log\n"
# exec {BASH_XTRACEFD}>>$log
# set -x

source download_nematus.sh

###
### prepare the data set
###
start=$(date +%s.%N)
infoecho "running prepdata.py ... \n"
python3 prepdata.py --config $config 2>&1 | tee -a $log
end=$(date +%s.%N)
diff=`show_time $end $start`
infoecho "prepdata: $diff \n"

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
infoecho "train_nmt.sh: "
checkconfig 'outdir'
checkconfig 'data'
checkconfig 'vocabsize_src'
checkconfig 'vocabsize_tgt'
infoecho "\n"
start=$(date +%s.%N)
CUDA_VISIBLE_DEVICES=$dev bash train_nmt.sh ${TRAIN[outdir]} ${TRAIN[data]} ${TRAIN[vocabsize_src]} ${TRAIN[vocabsize_tgt]} 2>&1 | tee -a $log

end=$(date +%s.%N)
diff=`show_time $end $start`
infoecho "train: $diff\n"

exit 0

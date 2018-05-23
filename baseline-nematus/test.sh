#!/bin/bash

source time.sh

today=`date +%Y-%m-%d.%H%M%S`
log=test.log.$today

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

source download_nematus.sh

###
### test nematus
###
function checkconfig()
{
    local var=$1
    if [ -z "${TEST[$var]}" ]; then
        echo "$0: cannot get config variable: $var for test_nmt.sh. Exit."
        exit 1
    fi
    printf "$var: ${TEST[$var]}, " | tee -a $log
}

eval "$(cat $config  | python ./ini2arr.py)"

printf "test_nmt.sh: " | tee -a $log
checkconfig 'outdir'
mkdir -p ${TEST[outdir]}
checkconfig 'modeldir'
checkconfig 'datadir'
checkconfig 'predict'
printf "\n" | tee -a $log
modelfiles=`python3 getmodels.py ${TEST[modeldir]}`
if [ -z "$modelfiles" ]; then
    echo "Cannot find model files in ${TEST[modeldir]}. Exit." | tee -a $log
    exit 1
else
    echo "model files: $modelfiles" | tee -a $log    
fi

start=$(date +%s.%N)
if [ -z "$dev" ]; then
    bash test_nmt.sh "$modelfiles" ${TEST[datadir]}/test.src.txt ${TEST[outdir]}/${TEST[predict]} ${TEST[modeldir]}/model.npz.json 2>&1 | tee -a $log
else
    CUDA_VISIBLE_DEVICES=$dev bash test_nmt.sh "$modelfiles" ${TEST[datadir]}/test.src.txt ${TEST[outdir]}/${TEST[predict]} ${TEST[modeldir]}/model.npz.json 2>&1 | tee -a $log
fi
end=$(date +%s.%N)
diff=`show_time $end $start`
echo "test: $diff" | tee -a $log

exit 0

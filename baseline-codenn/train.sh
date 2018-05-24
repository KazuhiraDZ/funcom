#!/bin/bash
source time.sh

today=`date +%Y-%m-%d.%H%M%S`
log=train.log.$today

###
### Parsing the command line arguments
###
read -r -d '' helpmsg <<EOM
Usage: $0
    -c          [required] set a config file
    -l          [optional] set the file name for log; default file name: $log
    -h          display this help message
EOM

passarg=true
while getopts ":c:l:h" opt; do
  case ${opt} in
    c )
	config=$OPTARG
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

source download_codenn.sh

###
### Parsing the config file
###
function checkconfig()
{
    local sec=$1
    local var=$2
    local secvar=$(eval echo $sec[$var])
    
    if [ -z "${!secvar}" ]; then
        echo "$0: cannot get config variable: $var in section $sec. Exit." | tee -a $log
        exit 1
    fi
    printf "[$sec] $var: ${!secvar}, " | tee -a $log
}

eval "$(cat $config  | python ./ini2arr.py)"
checkconfig 'CODENN' 'workdir'
checkconfig 'PREPDATA' 'outdir'
checkconfig 'PREPDATA' 'dataprep'
printf "\n" | tee -a $log

###
### PYTHON scripts: prepare the environment vairables
###
cwd=$(pwd)
export PYTHONPATH="${PYTHONPATH}:$cwd/codenn/src/"
export CODENN_DIR="$cwd/codenn"

workdir=${CODENN[workdir]}
if [[ "$DIR" = /* ]]; then
    export CODENN_WORK=$workdir
else
    # if the workdir is a relative path, change it to the absolute path
    export CODENN_WORK=$cwd/$workdir
fi
echo "PYTHONPATH: $PYTHONPATH, CODENN_DIR: ${CODENN_DIR}, CODENN_WORK: ${CODENN_WORK}" | tee -a $log

###
### prepare the data set
###
echo "running codenn/src/cpp/createParser ... " | tee -a $log
pushd ./codenn/src/cpp
bash createParser.sh 2>&1 | tee -a $cwd/$log
popd
echo "codenn/src/cpp/createParser.sh done" | tee -a $log

start=$(date +%s.%N)
echo "running prepdata.py ... " | tee -a $log
python3 ./prepdata.py --config $config 2>&1 | tee -a $log
end=$(date +%s.%N)
diff=`show_time $end $start`
echo "prepdata.py done: $diff" | tee -a $log

start=$(date +%s.%N)
echo "running parseData.py ... " | tee -a $log
python2 ./parseData.py --config $config 2>&1 | tee -a $log
end=$(date +%s.%N)
diff=`show_time $end $start`
echo "parseData.py done: $diff" | tee -a $log

###
### CODENN: buildData
###
missingfile=false
all=("$CODENN_WORK/test.data.cpp" "$CODENN_WORK/test.txt.cpp" "$CODENN_WORK/train.data.cpp" "$CODENN_WORK/train.txt.cpp" "$CODENN_WORK/valid.data.cpp" "$CODENN_WORK/valid.txt.cpp" "$CODENN_WORK/vocab.cpp" "$CODENN_WORK/vocab.data.cpp")
for i in "${all[@]}" ; do
    if [ ! -f $i ]; then
        missingfile=true
    fi
done 

if $missingfile ;then
    start=$(date +%s.%N)
    echo "running codenn/src/model/buildData.sh ... " | tee -a $log
    pushd ./codenn/src/model
    bash ./buildData.sh  2>&1 | tee -a $cwd/$log
    popd
    end=$(date +%s.%N)
    diff=`show_time $end $start`
    echo "codenn/src/model/buildData.sh done: $diff" | tee -a $log
else
    printf "existing files in ${CODENN_WORK}. \n!!!\n!!!**Skip** codenn/src/model/buildData.sh\n!!!\n" | tee -a $log
fi

###
### CODENN: train
###


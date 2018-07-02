#!/bin/bash
source util.sh

today=`date +%Y-%m-%d.%H%M%S`
log=train.log.$today

###
### Parsing the command line arguments
###
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
	dev=$((OPTARG+1)) # codenn (which uses lua torch) specify GPU device ids from 1
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

cwd=$(pwd)
log=$cwd/$log

infoecho "config file: $config, log file: $log \n"
# exec {BASH_XTRACEFD}>>$log
# set -x
source download_codenn.sh
source prepare_antlr.sh
source activate_lua.sh

###
### Parsing the config file
###
function checkconfig()
{
    local sec=$1
    local var=$2
    local secvar=$(eval echo $sec[$var])
    
    if [ -z "${!secvar}" ]; then
        infoecho "$0: cannot get config variable: $var in section $sec. Exit.\n"
        exit 1
    fi
    infoecho "[$sec] $var: ${!secvar}, "
}

eval "$(cat $config  | python ./ini2arr.py)"
checkconfig 'CODENN' 'workdir'
checkconfig 'PREPDATA' 'outdir'
checkconfig 'PREPDATA' 'dataprep'
checkconfig 'PREPDATA' 'lang'
checkconfig 'PREPDATA' 'maxlen_src'
checkconfig 'PREPDATA' 'maxlen_tgt'
checkconfig 'PREPDATA' 'batch_size'
checkconfig 'TRAIN' 'outdir'

infoecho "\n"
lang=${PREPDATA[lang]} # language of the data set
maxlen_src=${PREPDATA[maxlen_src]}
maxlen_tgt=${PREPDATA[maxlen_tgt]}
batch_size=${PREPDATA[batch_size]}

###
### prepare the environment vairables
###
export PYTHONPATH="${PYTHONPATH}:$cwd/codenn/src/"
export CODENN_DIR="$cwd/codenn"

workdir=$(absolutepath ${CODENN[workdir]} $cwd)
export CODENN_WORK=$workdir

infoecho "PYTHONPATH: $PYTHONPATH, CODENN_DIR: ${CODENN_DIR}, CODENN_WORK: ${CODENN_WORK}\n"

###
### prepare the data set
###
infoecho "language: $lang, running codenn/src/$lang/createParser ... \n"
pushd ./codenn/src/$lang
bash createParser.sh 2>&1 | tee -a $log
popd
infoecho "codenn/src/$lang/createParser.sh done\n"

start=$(date +%s.%N)
infoecho "running prepdata.py ... \n"
python3 ./prepdata.py --config $config 2>&1 | tee -a $log
retVal=$?
if [ $retVal -ne 0 ]; then
    error "Error in prepdata.py"
    end=$(date +%s.%N)
    diff=`show_time $end $start`
    infoecho "prepdata.py done: $diff\n"
    exit $retVal
fi
end=$(date +%s.%N)
diff=`show_time $end $start`
infoecho "prepdata.py done: $diff\n"

start=$(date +%s.%N)
infoecho "running parseData.py ... \n"
python2 ./parseData.py --config $config 2>&1 | tee -a $log
end=$(date +%s.%N)
diff=`show_time $end $start`
infoecho "parseData.py done: $diff\n"

###
### CODENN: buildData
###
missingfile=false
all=("$CODENN_WORK/test.data.$lang" "$CODENN_WORK/test.txt.$lang" "$CODENN_WORK/train.data.$lang" "$CODENN_WORK/train.txt.$lang" "$CODENN_WORK/valid.data.$lang" "$CODENN_WORK/valid.txt.$lang" "$CODENN_WORK/vocab.$lang" "$CODENN_WORK/vocab.data.$lang")
for i in "${all[@]}" ; do
    if [ ! -f $i ]; then
        missingfile=true
    fi
done 

if $missingfile ;then
    start=$(date +%s.%N)
    infoecho "running codenn/src/model/buildData.sh ... \n"
    pushd ./codenn/src/model
    bash ./buildData.sh $lang $maxlen_src $maxlen_tgt $batch_size 2>&1 | tee -a $log
    popd
    end=$(date +%s.%N)
    diff=`show_time $end $start`
    infoecho "codenn/src/model/buildData.sh done: $diff\n"
else
    infoecho "existing files in ${CODENN_WORK}. \n!!!\n!!!**Skip** codenn/src/model/buildData.sh\n!!!\n"
fi

###
### CODENN: train
###
modelout=$(absolutepath ${TRAIN[outdir]} $cwd)

if test "$(ls -A "$modelout")"; then
    infoecho "the output directory for model files is not empty.\n"
    infoecho "\n!!!\nSkip codenn/src/model/run.sh --> which means **skip** training!\n!!!\n"
    exit 0
else
    infoecho "the output directory does not exist or is empty. Good! Creating one.\n"
    mkdir -p $modelout
    start=$(date +%s.%N)
    infoecho "running codenn/src/model/main.lua ... \n"
<<<<<<< HEAD
<<<<<<< HEAD
    warning "model output dir: $modelout"
=======
>>>>>>> updated all the scripts to support a new type of input data file
=======
    warning "model output dir: $modelout"
>>>>>>> updated the training script and the test script. fixed a minor issue in the training script about the absolute paths.
    pushd ./codenn/src/model
    th ./main.lua -gpuidx $dev -language $lang -outdir $modelout -dev_ref_file $CODENN_WORK/valid.txt.$lang.ref
    popd
    end=$(date +%s.%N)
    diff=`show_time $end $start`
    infoecho "codenn/src/model/main.lua done: $diff \n"
fi


#!/bin/bash
source time.sh

today=`date +%Y-%m-%d.%H%M%S`
log=test.log.$today

RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

function infoecho(){ printf "$1" | tee -a $log; }
function warning(){ echo -e "${YELLOW}Warning: $1${NC}" | tee -a $log; }
function error(){ echo -e "${RED}Error: $1${NC}" | tee -a $log; }

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

cwd=$(pwd)
log=$cwd/$log

infoecho "config file: $config, log file: $log\n"
#exec {BASH_XTRACEFD}>>$log
#set -x
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
        infoecho "$0: cannot get config variable: $var in section $sec. Exit.\n"
        exit 1
    fi
    infoecho "[$sec] $var: ${!secvar}, "
}

eval "$(cat $config  | python ./ini2arr.py)"
checkconfig 'CODENN' 'workdir'
checkconfig 'TEST' 'modeldir'
checkconfig 'TEST' 'beamsize'
checkconfig 'TEST' 'predict'
checkconfig 'TEST' 'outdir'

function absolutepath(){
    local inpath=$1
    if [[ "$inpath" = /* ]]; then
        echo "$inpath"
    else
        echo "$cwd/$inpath"
    fi
}

workdir=$(absolutepath ${CODENN[workdir]})
predictout=$(absolutepath ${TEST[outdir]})
modeldir=$(absolutepath ${TEST[modeldir]})

mkdir -p $predictout
infoecho "\n"

###
### setting up environment paths
###
export CODENN_DIR="$cwd/codenn"
export CODENN_WORK=$workdir
infoecho "CODENN_DIR: ${CODENN_DIR}, CODENN_WORK: ${CODENN_WORK}\n"

###
### test
###
local encoders=($(ls -1v $modeldir/java.encoder.e*))
local decoders=($(ls -1v $modeldir/java.decoder.e*))

predictfile=${TEST[predict]}

if [ -f '$predictout/$predictfile' ]; then
    warning "$predictout/$predictfile exists! Move the file to: $predictout/${TEST[predict]}.old.$today"
    mv $predictout/${TEST[predict]} "$predictout/${TEST[predict]}".old.$today
else
    mkdir -p $predictout
    start=$(date +%s.%N)
    infoecho "running codenn/src/model/predict.lua ... \n"
    warning "output file: $predictout/$predictfile"
    pushd ./codenn/src/model
    th predict.lua -encoder ${encoders[-1]} -decoder ${decoders[-1]} -beamsize ${TEST[beamsize]} -gpuidx $dev -language java -outdir $predictout -outfile $predictfile
    popd
    end=$(date +%s.%N)
    diff=`show_time $end $start`
    infoecho "codenn/src/model/predict.lua done: $diff"
fi

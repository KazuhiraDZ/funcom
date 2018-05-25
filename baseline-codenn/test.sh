#!/bin/bash
source time.sh

today=`date +%Y-%m-%d.%H%M%S`
log=test.log.$today

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

echo "config file: $config, log file: $log" | tee -a $log
exec {BASH_XTRACEFD}>>$log
set -x
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
checkconfig 'TEST' 'modeldir'
checkconfig 'TEST' 'beamsize'
checkconfig 'TEST' 'datadir'
checkconfig 'TEST' 'predict'
checkconfig 'TEST' 'outdir'
mkdir -p ${TEST[outdir]}

printf "\n" | tee -a $log

# th predict.lua -encoder cpp.encoder -decoder cpp.decoder -beamsize $BEAMSIZE -gpuidx $dev -language cpp

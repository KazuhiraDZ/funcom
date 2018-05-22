#!/bin/bash
source time.sh

today=`date +%Y-%m-%d.%H%M%S`
log=run.log.$today

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

###
### Parsing the config file
###
function checkconfig()
{
    local sec=$1
    local var=$2
    local secvar=$(eval echo $sec[$var])
    
    if [ -z "${!secvar}" ]; then
        echo "$0: cannot get config variable: $var in section $sec. Exit."
        exit 1
    fi
    printf "[$sec] $var: ${!secvar}, " | tee -a $log
}

eval "$(cat $config  | python ./ini2arr.py)"

###
### prepare the environment vairables for codenn workflow
###
cwd=$(pwd)
export PYTHONPATH="${PYTHONPATH}:$cwd/codenn/src/"
export CODENN_DIR="$cwd/codenn"

workdir=${CODENN[workdir]}
if [[ "$DIR" = /* ]]; then
    export CODENN_WORK=$workdir
else
    export CODENN_WORK=$cwd/$workdir
fi
echo "PYTHONPATH: $PYTHONPATH, CODENN_DIR: ${CODENN_DIR}, CODENN_WORK: ${CODENN_WORK}"  | tee -a $log

###
### download codenn if necessary
###
function downloadcodenn()
{
    local codennurl=https://github.com/sjiang1/codenn.git
    echo "git submodule does not work... it's okay..."

    local codenndir=codenn
    if [ -d "$codenndir" ]; then
	echo "$codenndir already exists. if you would like to update the codenn copy, remove the folder first."
    else
	echo "we download from my fork..."
	# use my fork of codenn
	git clone --depth=1 $codennurl $codenndir
	rm -rf ${codenndir}/.git
    fi
}

git submodule init
submoduleflag=false
if [ $? -eq 0 ]
then
    git submodule update --remote
    if [ $? -eq 0 ]
    then
	pushd ./codenn
	git checkout master
	git pull
	popd
	submoduleflag=true
    fi
fi

if ! $submoduleflag ;then
    downloadcodenn
fi

###
### prepare the data set
###
pushd ./codenn/src/cpp
bash createParser.sh 2>&1 | tee -a $cwd/$log
popd

start=$(date +%s.%N)
python3 ./codenn/data/cpp/prepdata.py --config $cwd/$config 2>&1 | tee -a $log
end=$(date +%s.%N)
diff=`show_time $end $start`
echo "prepdata: $diff" | tee -a $log

start=$(date +%s.%N)
python2 ./codenn/data/cpp/parseData.py --config $cwd/$config 2>&1 | tee -a $log
end=$(date +%s.%N)
diff=`show_time $end $start`
echo "parsedata: $diff" | tee -a $log

# pushd ./codenn/src/model
# bash ./buildData.sh
# popd

# printf "train.sh: " | tee -a $log
# checkconfig 'CODENN' 'workdir'
# printf "\n" | tee -a $log
exit 0

# start=$(date +%s.%N)
# python3 prepdata.py --config $config 2>&1 | tee -a $log
# end=$(date +%s.%N)
# diff=`show_time $end $start`
# echo "prepdata: $diff" | tee -a $log

# function checkconfig()
# {
#     local var=$1
#     if [ -z "${TRAIN[$var]}" ]; then
#         echo "$0: cannot get config variable: $var for train.sh. Exit."
#         exit 1
#     fi
#     printf "$var: ${TRAIN[$var]}, "
# }

# eval "$(cat $config  | python ./ini2arr.py)"
# printf "train.sh: "
# checkconfig 'outdir'
# checkconfig 'data'
# checkconfig 'vocabsize_src'
# checkconfig 'vocabsize_tgt'
# printf "\n"
# start=$(date +%s.%N)
# bash train.sh ${TRAIN[outdir]} ${TRAIN[data]} ${TRAIN[vocabsize_src]} ${TRAIN[vocabsize_tgt]} | tee -a $log
# end=$(date +%s.%N)
# diff=`show_time $end $start`
# echo "prepdata: $diff" | tee -a $log

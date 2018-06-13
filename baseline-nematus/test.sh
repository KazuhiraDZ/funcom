#!/bin/bash
source time.sh

today=`date +%Y-%m-%d.%H%M%S`
log=test.log.$today

RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

function infoecho(){ printf $1 | tee -a $log; }
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
### test nematus
###
function checkconfig()
{
    local var=$1
    if [ -z "${TEST[$var]}" ]; then
        error "$0: cannot get config variable: $var for test_nmt.sh. Exit."
        exit 1
    fi
    infoecho "$var: ${TEST[$var]}, "
}

eval "$(cat $config  | python ./ini2arr.py)"

infoecho "test_nmt.sh: "
checkconfig 'outdir'
mkdir -p ${TEST[outdir]}
checkconfig 'modeldir'
checkconfig 'datadir'
checkconfig 'predict'
infoecho "\n"
modelfiles=`python3 getmodels.py ${TEST[modeldir]}`
if [ -z "$modelfiles" ]; then
    error "Cannot find model files in ${TEST[modeldir]}. Exit."
    exit 1
else
    infoecho "model files: $modelfiles\n"    
fi

start=$(date +%s.%N)
function runtest(){
    local testfile=$1
    infoecho "runtest: $testfile\n"
    CUDA_VISIBLE_DEVICES=$dev bash test_nmt.sh "$modelfiles" $testfile ${testfile}.predict ${TEST[modeldir]}/model.npz.json 2>&1 | tee -a $log    
}

mkdir -p ${TEST[datadir]}/testsplitfiles
if [ "$(ls -A ${TEST[datadir]}/testsplitfiles)" ]; then
    warning "the testsplitfiles are not empty. Will use the existing test files AND prediction files!"
else
    split -a 4 -d -l 3000 ${TEST[datadir]}/test.src.txt ${TEST[datadir]}/testsplitfiles/test.src.txt_
fi

if [ -f ${TEST[outdir]}/${TEST[predict]} ]; then
    mv ${TEST[outdir]}/${TEST[predict]} "${TEST[outdir]}/${TEST[predict]}".old.$today
fi
rm -f "${TEST[outdir]}/${TEST[predict]}".duplicate

ls -v `find ${TEST[datadir]}/testsplitfiles/ -name "test.src.txt_[0-9][0-9][0-9][0-9]"` | while read line; do
    filename=$line
    infoecho "running nematus on $filename ...\n"
    if [ -s ${filename}.predict ]; then
    	warning "Skip running nematus. Using the existing ${filename}.predict"
    	cat ${filename}.predict >> ${TEST[outdir]}/${TEST[predict]}
    	cat $filename >> "${TEST[outdir]}/${TEST[predict]}".duplicate
    	continue
    fi
    
    runtest "$filename";
    if [ -s ${filename}.predict ]
    then
    	cat ${filename}.predict >> ${TEST[outdir]}/${TEST[predict]}
    	cat $filename >> "${TEST[outdir]}/${TEST[predict]}".duplicate
    else
    	warning "Nematus did not generate prediciton file for $filename. Try splitting it again."
	rm -f ${filename}_*
    	split -a 4 -d -l 100 $filename ${filename}_
    	for smaller_filename in ${filename}_*; do
    	    runtest "$smaller_filename"
    	    if [ -s ${smaller_filename}.predict ]
    	    then
    		cat ${smaller_filename}.predict >> ${TEST[outdir]}/${TEST[predict]}
    		cat ${smaller_filename} >> "${TEST[outdir]}/${TEST[predict]}".duplicate
    	    else
    		error "Nematus still cannot generate predictions for $smaller_filename "
    		exit 1
    	    fi
    	done
    fi
done

end=$(date +%s.%N)
diff=`show_time $end $start`
infoecho "test: $diff\n"

exit 0

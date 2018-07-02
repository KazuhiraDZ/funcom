#!/bin/bash

function absolutepath(){
    local inpath=$1
    local cwd=$2
    if [[ "$inpath" = /* ]]; then
        echo "$inpath"
    else
        echo "$cwd/$inpath"
    fi
}

# copied from: https://stackoverflow.com/questions/12199631/convert-seconds-to-hours-minutes-seconds
function show_time () {
    diff=$(echo "$1 - $2" | bc)
    num=${diff%.*}
    min=0
    hour=0
    day=0
    if((num>59));then
	((sec=num%60))
	((num=num/60))
	if((num>59));then
	    ((min=num%60))
	    ((num=num/60))
	    if((num>23));then
		((hour=num%24))
		((day=num/24))
	    else
		((hour=num))
	    fi
	else
	    ((min=num))
	fi
    else
	((sec=num))
    fi
    echo "${day}d ${hour}h ${min}m ${sec}s"
}

RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

function infoecho(){ printf "$1" | tee -a $log; }
function warning(){ echo -e "${YELLOW}Warning: $1${NC}" | tee -a $log; }
function error(){ echo -e "${RED}Error: $1${NC}" | tee -a $log; }

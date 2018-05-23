#!/bin/bash

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

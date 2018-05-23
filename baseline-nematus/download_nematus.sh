#!/bin/bash

###
### download nematus if necessary
###
function downloadnematus()
{
    echo "git submodule does not work... it's okay..."

    local nematusdir=nematus-tensorflow
    if [ -d "$nematusdir" ]; then
	echo "nematus-tensorflow already exists. if you would like to update the nematus copy, remove the folder first." | tee -a $log
    else
	echo "we download from my fork..." | tee -a $log
	# use my fork of nematus, in case there is any major change in the original nematus repo
	git clone --depth=1 --branch=tensorflow https://github.com/sjiang1/nematus.git $nematusdir
	rm -rf ${nematusdir}/.git
    fi
}

git submodule init
submoduleflag=false
if [ $? -eq 0 ]
then
    git submodule update
    if [ $? -eq 0 ]
    then
	submoduleflag=true
    fi
fi

if ! $submoduleflag ;then
    downloadnematus
fi

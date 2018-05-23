#!/bin/bash

###
### download nematus if necessary
###
nematusdir=nematus-tensorflow
function downloadnematus()
{
    echo "git submodule does not work... it's okay..."

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
    git submodule update --remote
    if [ $? -eq 0 ]
    then
	pushd ./$nematusdir
	git checkout master
	git pull
	popd
	submoduleflag=true
    fi
fi

if ! $submoduleflag ;then
    downloadnematus
fi

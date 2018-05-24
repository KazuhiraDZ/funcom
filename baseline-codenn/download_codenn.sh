#!/bin/bash

###
### download codenn if necessary
###
function downloadcodenn()
{
    local codennurl=https://github.com/sjiang1/codenn.git
    echo "git submodule does not work... it's okay..." | tee -a $log

    local codenndir=codenn
    if [ -d "$codenndir" ] && [ -f "$codenndir/src/cpp/CppTemplate.py" ]; then
	echo "$codenndir already exists. if you would like to update the codenn copy, remove the folder first." | tee -a $log
    else
	echo "we download from my fork..." | tee -a $log
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


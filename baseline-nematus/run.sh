#!/bin/bash

tensorflowurl=https://github.com/EdinburghNLP/nematus/archive/tensorflow.zip

function downloadnematus()
{
    echo "git submodule does not work... it's okay..."

    nematusdir=nematus-tensorflow
    if [ -d "$nematusdir" ]; then
	echo "nematus-tensorflow already exists. if you would like to update the nematus copy, remove the folder first."
    else
	echo "we download from my fork..."
	# use my fork of nematus, in case there is any major change in the original nematus repo
	git clone --depth=1 --branch=tensorflow https://github.com/sjiang1/nematus.git $nematusdir
	rm -rf ${nematusdir}/.git	
    fi
}


git submodule init
if [ $? -eq 0 ]
then
    git submodule update
    if [ $? -ne 0 ]
    then
	downloadnematus
    fi
else
    downloadnematus
fi

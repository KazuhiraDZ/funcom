#!/bin/bash

source util.sh

cwd=$(pwd)
if [ ! -f antlr-4.5.3-complete.jar ]; then
    echo "downloading antlr jar ..."
    curl -O http://www.antlr.org/download/antlr-4.5.3-complete.jar
    export CLASSPATH=.:$cwd/antlr-4.5.3-complete.jar:$CLASSPATH
fi

if python2 -c "import antlr4" &> /dev/null; then
    echo 'found antlr4 in python2 modules'
else
    error "Antlr4 is not found in python2 module. Install this by: python2 -m pip install 'antlr4-python2-runtime>=4.5,<4.6'"
    exit 1
fi

#!/bin/bash

ref=$1
pred=$2

function checkmodule(){
    local module=$1
    
    python3 -c "import $module"
    if [ $? -ne 0 ]; then
        echo "python3: Missing $module... Installing $module ..."
        python3 -m pip install $module
    fi
}

checkmodule nltk
checkmodule argparse

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
python3 $DIR/nltk_bleu.py $ref $pred 2>/dev/null
if [ $? -ne 0 ]; then
    echo "usage: $0 reference_file prediction_file"
fi

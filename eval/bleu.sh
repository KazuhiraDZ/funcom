#!/bin/bash

ref=$1
pred=$2

source ./configure.sh
$mosesdecoder/scripts/generic/multi-bleu.perl $ref < $pred

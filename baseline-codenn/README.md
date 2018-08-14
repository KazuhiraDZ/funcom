# funcom - baseline-codenn

Scripts for running [codenn](https://github.com/sjiang1/codenn) on our data sets.

### Prerequisites
python2, python3, pip, ntlk, tensorflow and keras in python2 and python3 \
configparser, argparse modules \
tested on Ubuntu 16 and Debian GNU/Linux 8.10

### Default configuration and current results
08/07/2018 \
(informal, manually checked) Peak cpu memory usage: 12GB \
(informal, manually checked) Peak gpu memory usage: 9GB \
so, don't run it on bishop in the future (just to be safe and avoid extra work).

### Notes
The test set entries are sorted based on fid in the [dataset](https://github.com/mcmillco/funcom/tree/master/alpha/dataprep). (so the results are sorted, too)

### Overview of this baseline

Usage:
1) modify the codenn.ini based on the above workflow image.\
   most likely, you need to change ```dataprep``` under ```[PREPDATA]``` to the folder that has alldata.pkl, comstokenizer.pkl and datstokenizer.pkl
2) ```bash train.sh -c codenn.ini -d 0``` \
   run ```bash train.sh -h``` to see other options \
   Note that GPU device id starts from 0.

3) ```bash test.sh -c codenn.ini -d 1```

### Obsolete
06/26/2018 \
Input data file: /scratch/funcom_data/: fundats-j1.pkl, coms.test, coms.train, coms.val, dats.test, dats.train, dats.val \
input vocab size: unknown \
output vocab size: unknown \
threshold for vocab: 2 (if the frequency of a word is smaller or equal to 2, the word is excluded from the vocabulary) \
maxlen of source sequences: 100 \
maxlen of target sequences: 13 \
actual train/dev/test sets sizes: 873223/53415/38521 (codenn has its own preprocessing for raw code) \
batch size: 80 \
training epochs: 137 \
training time: 4d 6h 50m 46s \
model number: 1 (use one model for testing) \
beam size: 1 (no beam search in testing) \
testing time: 0d 0h 8m 13s \
BLEU: 6.03 (12.83, 7.19, 4.48, 3.19)

models are on ash: ```/scratch/funcom_old/baseline-codenn/workdir_java03/models/``` \
results are on ash: ```/scratch/funcom_old/baseline-codenn/workdir_java03/test-beam1-model1/```

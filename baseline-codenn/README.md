# funcom - baseline-codenn

Scripts for running [codenn](https://github.com/sjiang1/codenn) on our data sets.

### Prerequisites
python2, python3, tensorflow and keras in python2 and python3 \
configparser, argparse modules \
tested on Ubuntu 16 and Debian GNU/Linux 8.10

### Notes
The test set entries are sorted based on fid in the [dataset](https://github.com/mcmillco/funcom/tree/master/alpha/dataprep). (so the results are sorted, too)

### Overview of this baseline

Usage:
1) modify the codenn.ini based on the above workflow image.\
   most likely, you need to change ```dataprep``` under ```[PREPDATA]``` to the folder that has alldata.pkl, comstokenizer.pkl and datstokenizer.pkl
2) ```bash train.sh -c codenn.ini -d 1``` \
   run ```bash train.sh -h``` to see other options \
   **Note that GPU device id starts from 1**

3) ```bash test.sh -c codenn.ini -d 1```
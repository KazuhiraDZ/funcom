# funcom - baseline-nematus

Scripts for running [nematus-tensorflow](https://github.com/EdinburghNLP/nematus/tree/tensorflow) on our data sets.

### Prerequisites
python3, python2, keras, tensorflow in python3 & python2\
tested on Ubuntu 16 and Debian GNU/Linux 8.10

### Overview of this baseline
<img src="workflow.png" width="400">

**Note** we assume the word_index in the tokenizers (comstokenizer.pkl, datstokenizer.pkl) are fitted to the train data set.

Usage:
1) modify the nematus.ini based on the above workflow image.\
   most likely, you need to change ```dataprep``` to point to the folder that has alldata.pkl, comstokenizer.pkl and datstokenizer.pkl
2) ```bash train.sh -c nematus.ini```\
   ```bash train.sh -h``` to display other options
3) ```bash test.sh -c nematus.ini```\
   ```bash test.sh -h``` to display other options

Notes:
1) this nematus does not support pre-trained embedding. (compared to the alpha version where we use a pre-trained embedding)
2) like the alpha version, **for now**, we use a subset from the training set as the valid set
3) uses the same vocab size for tgt sequences as the alpha model: for tgt sequences, 10449
4) different from the alpha model, for src sequences, we use 50k instead of all the words (too big for nematus..)
5) **for testing**: gpu memory is often not large enough for testing the entire test set. Try something like:\
```split -a 3 -d -l 300 data/test.src.txt data/test.src.split/test.src.part``` to split the test file into smaller pieces.

Files:
1) **prepdata.py**: preparing the data files for nematus (based on the files generated in [alpha/dataprep](https://github.com/mcmillco/funcom/tree/master/alpha/dataprep))
2) **run.log**: the log of commands used for running the scripts
3) **train.sh**: the script to run nematus ([nematus-tensorflow](https://github.com/EdinburghNLP/nematus/tree/tensorflow) is needed)
    * change nematus path if needed
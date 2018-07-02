# funcom - baseline-nematus

Scripts for running [nematus-tensorflow](https://github.com/EdinburghNLP/nematus/tree/tensorflow) on our data sets.

### Prerequisites
python3, python2, keras, tensorflow in python3 & python2\
tested on Ubuntu 16 and Debian GNU/Linux 8.10 with **/scratch/funcom_data**

### Default configuration and current results
input files: /scratch/funcom_data/: coms.test, coms.train, coms.val, dats.test, dats.train, dats.val \
input vocab size: 50000 \
target vocab size: 44707 \
maxlen of target sequences: 13 \
maxlen of source sequences: 100 \
stopping criterion: the default stopping criterion Nematus use (early-stopping-method) \
training time: 0d 11h 35m 30s \
number of epochs: 14 \
beam width = 1 (disable the beam search) \
model number = 1 (no ensemble test) \
testing time (tested on bishop): 0d 5h 39m 24s \
BLEU score: 23.90 (48.17, 27.66, 17.98, 13.62) \
models are on ash: ```/scratch/funcom_old/baseline-nematus/workdir_java03/models/``` \
results are on ash: ```/scratch/funcom_old/baseline-nematus/workdir_java03/test-beam1-model1```


### Overview of this baseline
<img src="workflow.png" width="400">

**Note1** we assume the word indices in the tokenizers (comstokenizer.pkl, datstokenizer.pkl) are sorted based on the frequencies of the words.

**Note2** we assume you have the ```/scratch/funcom/sourceme.sh``` (set up the python path for Python Class: ```Tokenizer```)

**Note3** the test set entries are sorted based on fid in the [dataset](https://github.com/mcmillco/funcom/tree/master/alpha/dataprep). (so the predictions are sorted, too.)

**Usage**:
1) modify the nematus.ini based on the above workflow image.\
   most likely, you need to change the following entries:
   * ```dataprep``` under ```[PREPDATA]```: point to the folder that has alldata.pkl, comstokenizer.pkl and datstokenizer.pkl
   * ```predict``` under ```[TEST]```: the output file name for the predictions
2) ```bash train.sh -c nematus.ini```\
   ```bash train.sh -h``` to display other options
3) ```bash test.sh -c nematus.ini```\
   ```bash test.sh -h``` to display other options

Extra notes:
1) this nematus does not support pre-trained embedding. (compared to the alpha version where we use a pre-trained embedding)
2) like the alpha version, **for now**, we use a subset from the training set as the valid set
3) uses the same vocab size for tgt sequences as the alpha model: for tgt sequences, 10449
4) different from the alpha model, for src sequences, we use 50k instead of all the words (too big for nematus..)
5) **for testing**: gpu memory is often not large enough for testing the entire test set. In the script, we split the test file into smaller pieces. First, we split the file into small files of 3K lines. It turned out, for some small files, we still got "out of memory" errors. For those files, we split them into smaller files of 100 lines.

### old setting
input vocab size: 50000 \
target vocab size: 10449 \
stopping criterion: the default stopping criterion Nematus use (early-stopping-method) \
number of epochs: 63 \
batch size: 80 \
training time: 3d 6h 9m 21s \
beam width: 12 (beam search in testing)
model number: 4 (ensemble testing)
BLEU score: on the data set preprocessed by an obsolete filter \
BLEU4: 44.60 (53.69, 44.89, 41.77, 39.32)

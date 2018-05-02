# funcom - baseline-nematus

Scripts for running [nematus-tensorflow](https://github.com/EdinburghNLP/nematus/tree/tensorflow) on our data sets.

Notes:
1) this nematus does not support pre-trained embedding. (compared to the alpha version where we use a pre-trained embedding)
2) like the alpha version, **for now**, we use a subset from the training set as the valid set
3) uses the same vocab size for tgt sequences as the alpha model: for tgt sequences, 10449
4) different from the alpha model, for src sequences, we use 50k instead of all the words (too big for nematus..)

Files:
1) **prepdata.py**: preparing the data files for nematus (based on the files generated in [alpha/dataprep](https://github.com/mcmillco/funcom/alpha/dataprep))
2) **run.log**: the log of commands used for running the scripts
3) **train.sh**: the script to run nematus ([nematus-tensorflow](https://github.com/EdinburghNLP/nematus/tree/tensorflow) is needed)
    * change nematus path if needed
# funcom - alpha
A modified image-captioning model ([LemonATsu/Keras-Image-Caption](https://github.com/LemonATsu/Keras-Image-Caption)) for generating NL sequences from code sequences.

1) **bleu.sh**: calculating bleu score using [moses](https://github.com/moses-smt/mosesdecoder/tree/RELEASE-4.0) scripts.
2) **configure.sh**: env setup that you don't need to care
3) **eval.py**: python code for run a trained model
4) **model.py**: the actual model
5) **playground.py**: python code for train a model
6) **test.sh**: test a trained model using ```eval.py```
7) **dataprep/**: see README of the folder
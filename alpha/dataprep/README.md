# funcom - alpha - data preparation

author: [mcmillco](https://github.com/mcmillco) \
date: April 2018

**Note**: all the pkl files are not included in this git repository.

1) **comget.py**: Reads from database and archive on disk (on ```ash```) to create funcoms.pkl and
   fundats.pkl.  The "coms" refers to the block comments above functions.  The
   "dats" refers to the contents of the function.  The pkl files are
   dictionaries where the key is the function ID from the database.    

2) **preprocessall.py**: Reads the pkl files and creates paired dat/text files.  By
   paired I mean that e.g. line 50 in funcoms.dat is the preprocessed comment
   for the function body at line 50 in fundats.dat.  These dat files include
   the function ID from the database on the left side of the colon in a line.

3) **tokenizeall.py**: Reads the funcoms.dat and fundats.dat files and creates a
   Keras tokenizer.  Stores the tokenizer to tokenizer.pkl.

4) **splitbyproj.py**: Reads the dat files and splits the data into a training and
   testing set.  This split is by project, meaning that all functions from a
   project are in either the training or test set -- there is no case where a
   function in the training set is in the same project as a function from the
   test set.  Default is that 10% of the projects will be in the test set.
   Note that due to differing numbers of functions per project, the test set
   may have slightly more or less than 10% of the functions.  Outputs:
   funcoms.test.dat, funcoms.train.dat, fundats.test.dat, and fundats.train.dat
   These output files do not have the function ID.  However, since the random
   seed is fixed (to 1337), rerunning the script should produce the same
   results, so the links to function IDs can be recovered.

   Also uses the tokenizer to create sequences, and saves everything in a zilla
   dictionary of the form:
   ```
   alldata['coms_raw']          non-preprocessed comments
   alldata['dats_raw']          non-preprocessed fun. contents
   alldata['coms_test_pp']      preprocessed comments       text        test
   alldata['dats_test_pp']      preprocessed fun. contents  text        test
   alldata['coms_train_pp']     preprocessed comments       text        train
   alldata['dats_train_pp']     preprocessed fun. contents  text        train
   alldata['coms_test_seqs']    preprocessed comments       tokenized   test
   alldata['dats_test_seqs']    preprocessed fun. contents  tokenized   test
   alldata['coms_train_seqs']   preprocessed comments       tokenized   train
   alldata['dats_train_seqs']   preprocessed fun. contents  tokenized   train
   alldata['testpids']          list of test project ids
    ```
   The dictionaries all have a key is the function id.  E.g.
    ```
   coms_raw = alldata['coms_raw']
   coms_raw[fid]
    ```

   The function id is the same one from the database.\
   The dictionary ```alldata``` is saved using pickle protocol 2 for compatibility.

5) **vocabmaker.py**: Creates vocab json files in format used by nematus.  Creates
   separate vocabs for comments and function contents data.  Default vocab
   size is 30000, changable as max_words variable in script.
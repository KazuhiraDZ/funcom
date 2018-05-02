from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
#from sklearn import preprocessing
import pickle
import numpy as np

print('creating complete word list')

fcom = open('funcoms.dat', 'r')
fdat = open('fundats.dat', 'r')



allwords = set()

i = 0
for com in fcom.readlines():
    allwords.add(com)
    i += 1
    if i % 10000 == 0:
        print(i)

print('tokenizing coms...')

# top 10449 words are those that occur at least 100 times
#tok = Tokenizer(num_words = 10449)
tok = Tokenizer(oov_token="UNK")
tok.fit_on_texts(allwords)

print('writing coms tokenizer to disk')

pickle.dump(tok, open('comstokenizer.pkl', 'wb'))



allwords = set()

i = 0
for dat in fdat.readlines():
    allwords.add(dat)
    i += 1
    if i % 10000 == 0:
        print(i)

print('tokenizing dats...')

tok = Tokenizer(oov_token="UNK")
tok.fit_on_texts(allwords)

print('writing dats tokenizer to disk')

pickle.dump(tok, open('datstokenizer.pkl', 'wb'))

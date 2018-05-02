import MySQLdb
import pickle
import re
import json
import operator
#from bs4 import UnicodeDammit
from cachedict import cachedict

max_words = 30000

fcom = open("funcoms.dat", "r")
fdat = open("fundats.dat", "r")

comvocab = dict()
datvocab = dict()

for (com, dat) in zip(fcom.readlines(), fdat.readlines()):
    
    if com == '':
        continue
    
    if dat == '':
        continue

    #allwords = com + " " + dat
    
    for word in com.split():
        if word in comvocab:
            comvocab[word] += 1
        else:
            comvocab[word] = 1

    for word in dat.split():
        if word in datvocab:
            datvocab[word] += 1
        else:
            datvocab[word] = 1

comlimvocab = dict()
comlimvocab['eos'] = 1000000
comlimvocab['UNK'] = 1000000

datlimvocab = dict()
datlimvocab['eos'] = 1000000
datlimvocab['UNK'] = 1000000

i = 2
for word, count in sorted(comvocab.items(), key=operator.itemgetter(1), reverse=True):
    if i <= max_words:
        comlimvocab[word] = i
    i += 1

i = 2
for word, count in sorted(datvocab.items(), key=operator.itemgetter(1), reverse=True):
    if i <= max_words:
        datlimvocab[word] = i
    i += 1

comoutfile = open("vocab.%d.com.json" % max_words, 'w')
json.dump(comlimvocab, comoutfile, indent=2, ensure_ascii=False)

datoutfile = open("vocab.%d.dat.json" % max_words, 'w')
json.dump(datlimvocab, datoutfile, indent=2, ensure_ascii=False)

fdat.close()
fcom.close()
comoutfile.close()
datoutfile.close()


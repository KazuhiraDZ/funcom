import MySQLdb
import pickle
#from bs4 import UnicodeDammit
from cachedict import cachedict

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import random
random.seed(1337)

try:
	fidproj = pickle.load(open("fidproj.pkl", "rb"))
except Exception as ex:
	conn = MySQLdb.connect(host = "localhost", user = "ports_20k", passwd = "s3m3rU", db = "debian_25k")
	cursor = conn.cursor()

	query = "select F.id, I.project from functionalunits F, files I where F.fileid = I.fileid and I.project"

	cursor.execute(query)
	res = cursor.fetchall()

	fidproj = dict()

	for fid, pid in res:
		fidproj[int(fid)] = int(pid)

	pickle.dump(fidproj, open("fidproj.pkl", "wb"))

pids = set()

for fid in fidproj:
	pids.add(fidproj[fid])

testpids = set()
trainpids = set()

for pid in pids:
	rand = random.randint(1, 10)
	if rand == 1:
		testpids.add(pid)
	else:
		trainpids.add(pid)

fcom = open('funcoms.dat', 'r')
fdat = open('fundats.dat', 'r')

#fcom1 = open('funcoms.test.dat', 'w')
#fcom2 = open('funcoms.train.dat', 'w')
#fdat1 = open('fundats.test.dat', 'w')
#fdat2 = open('fundats.train.dat', 'w')

fcom1 = dict()
fcom2 = dict()
fdat1 = dict()
fdat2 = dict()
fcom1seqs = dict()
fcom2seqs = dict()
fdat1seqs = dict()
fdat2seqs = dict()

comstok = pickle.load(open('comstokenizer.pkl', 'rb'), encoding="UTF-8")
datstok = pickle.load(open('datstokenizer.pkl', 'rb'), encoding="UTF-8")
i = 0

for com, dat in zip(fcom.readlines(), fdat.readlines()):
    (fid, com) = com.split(': ')
    (fid, dat) = dat.split(': ')
    fid = int(fid)

    if fidproj[fid] in testpids:
        fcom1[fid] = com
        fdat1[fid] = dat
		#fcom1.write(com)
		#fdat1.write(dat)
    else:
        fcom2[fid] = com
        fdat2[fid] = dat
		#fcom2.write(com)
		#fdat2.write(dat)
		
    i += 1
    if i % 10000 == 0:
        print(i)

def toseq_preserve_order(d, tok, num_words=None, max_seq_len=None, pad=False):
    ret = dict()
    fl = list()
    for fid in sorted(d.keys()):
        fl.append(d[fid])
    tok.num_words = num_words
    seqs = tok.texts_to_sequences(fl) # note: this REMOVES the >num_words words, it doesn't replace them with UNK
    if pad:
        seqs = pad_sequences(seqs, maxlen=max_seq_len, padding='post', truncating='post')
    for fid, seq in zip(sorted(d.keys()), seqs):
        #del seq[50:] # truncate sequences to a maximum length of 50
        if max_seq_len:
            seq = seq[:max_seq_len]
        ret[fid] = seq
    return ret

print("1")
fcom1seqs = toseq_preserve_order(fcom1, comstok, num_words=10449, max_seq_len=50)
print("2")
fdat1seqs = toseq_preserve_order(fdat1, datstok, max_seq_len=50, pad=True)
print("3")
fcom2seqs = toseq_preserve_order(fcom2, comstok, num_words=10449, max_seq_len=50)
print("4")
fdat2seqs = toseq_preserve_order(fdat2, datstok, max_seq_len=50, pad=True)

fcom.close()
fdat.close()
#fcom1.close()
#fcom2.close()
#fdat1.close()
#fdat2.close()

funcomsraw = pickle.load(open("funcoms.pkl", "rb"))
fundatsraw = pickle.load(open("fundats.pkl", "rb"))

alldata = dict()
alldata['coms_raw'] = funcomsraw
alldata['dats_raw'] = fundatsraw
alldata['coms_test_pp'] = fcom1
alldata['dats_test_pp'] = fdat1
alldata['coms_train_pp'] = fcom2
alldata['dats_train_pp'] = fdat2
alldata['coms_test_seqs'] = fcom1seqs
alldata['dats_test_seqs'] = fdat1seqs
alldata['coms_train_seqs'] = fcom2seqs
alldata['dats_train_seqs'] = fdat2seqs
alldata['testpids'] = testpids

pickle.dump(alldata, open('alldata.pkl', 'wb'), protocol=2)

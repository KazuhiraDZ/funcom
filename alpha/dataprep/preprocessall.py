import MySQLdb
import pickle
import re
#from bs4 import UnicodeDammit
from cachedict import cachedict

def preprocess(text):
    
    # remove non-word chars
    text = re.sub(r'[^\w]', ' ', text)
    
    # split camel case
    text = re.sub(r'([A-Z][a-z])', r' \1', text)
    
    # condense multiple whitespace to one
    text = ' '.join(text.split())
    
    text = text.lower()
    
    return text

funcoms = pickle.load(open("funcoms.pkl", "rb"))
fundats = pickle.load(open("fundats.pkl", "rb"))

fcom = open("funcoms.dat", "w")
fdat = open("fundats.dat", "w")

for fid in funcoms.keys():

    com = funcoms[fid]
    dat = fundats[fid]
    
    if com == '':
        continue
    
    if dat == '':
        continue

    com = preprocess(com)
    dat = preprocess(dat)

    fcom.write("%d: %s\n" % (fid, com))
    fdat.write("%d: %s\n" % (fid, dat))
    
    if(fid % 1000 == 0):
        print(fid)

fdat.close()
fcom.close()

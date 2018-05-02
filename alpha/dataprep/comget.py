import MySQLdb
import pickle
#from bs4 import UnicodeDammit
from cachedict import cachedict

print("accessing function list from database...")

conn = MySQLdb.connect(host = "localhost", user = "ports_20k", passwd = "s3m3rU", db = "debian_25k")

query = "select F.id, F.linebegin, F.lineend, I.name from functionalunits F, files I where F.fileid = I.fileid"

cursor = conn.cursor()
cursor.execute(query)
res = cursor.fetchall()

memfiles = cachedict(20000)
commentcount = 0
funcount = 0

funcoms = dict() # function comments
fundats = dict() # function data/content

print("processing functions...")

for fid, linebegin, lineend, filename in res:
    #print(fid, linebegin, lineend, filename)
    try:
        if filename not in memfiles:
            with open(filename, errors='replace') as x: f = x.readlines()
            memfiles[filename] = f
    except FileNotFoundError as ex:
        print("warning: file not found: " + filename)
        continue
    except UnicodeDecodeError as ex:
        print("warning: unfixable unicode error in %s" % (filename))
        continue

    commentstart = -1
    # if line preceeding function contains a comment block end
    try:
        if '*/' in memfiles[filename][linebegin-2]:
            j = linebegin - 2
            while '/*' not in memfiles[filename][j]:
                j -= 1
            commentstart = j
    except (IndexError, KeyError) as ex:
        continue # ignore functions without comments
    
    funcoms[fid] = str()
    fundats[fid] = str()
    
    if commentstart != -1:
        commentcount += 1
        for i in range(commentstart, linebegin-1):
            funcoms[fid] += memfiles[filename][i]
            
    for i in range(linebegin-1, lineend):
        try:
            fundats[fid] += memfiles[filename][i]
        except KeyError as ex:
            print("warning: line %d not found in %s" % (linebegin, filename))
            continue

    #print(funcoms[fid])
    #print(fundats[fid])
    
    funcount += 1
    if(funcount % 10000 == 0):
        print(funcount, commentcount)

print("number of functions: %d" % (funcount))
print("number of comments:  %d" % (commentcount))

pickle.dump(funcoms, open("funcoms.pkl", "wb"))
pickle.dump(fundats, open("fundats.pkl", "wb"))

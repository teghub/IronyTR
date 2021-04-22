#

# For citation, please check:
# https://github.com/teghub/IronyTR

# This code is a helper for extracting graph similarity scores.
# You may need to create the missing files.

data = open("../data/lemmatized-whole.txt", "r")
binary = open("../data/binary-whole.txt", "r")
emotes = open("../lookup/emoji-emoticons.txt", "r")
lookup = open("../lookup/lookup.txt", "r")
graphscores = open("../data/graph-scores.txt", "a+")

emos = [line.rstrip('\n') for line in emotes]
lookuplist = [line.rstrip('\n') for line in lookup]
ll = []
for l in lookuplist:
    linelist = l.split(" ")
    ll.append(linelist[0])

allowed = ll + emos

sentences = [line.rstrip('\n') for line in data]
b = [line.rstrip('\n') for line in binary]

nn = 0
nodemap = {}

for s in sentences:
    swl = s.split()
    for w in swl:
        if w in allowed:
            if w not in nodemap:
                nodemap[w] = nn
                nn += 1
ig = []
ng = []
k = 0
kk = 0
for k in range(nn):
    rowig = []
    rowng = []
    for kk in range(nn):
        rowig.append(0)
        rowng.append(0)
    ig.append(rowig)
    ng.append(rowng)

# create class graphs
bi = 0
for s in sentences:
    swl = s.split()
    dellist = []
    for w in swl:
        if w not in allowed:
            dellist.append(w)
    for dw in dellist:
        swl.remove(dw)
    slen = len(swl)-3
    rlen = len(swl)
    if rlen > 3:
        js = 0
        for js in range(slen):
            if b[bi] == '0':
                ng[nodemap[swl[js]]][nodemap[swl[js+1]]] += 1
                ng[nodemap[swl[js]]][nodemap[swl[js+2]]] += 1
                ng[nodemap[swl[js]]][nodemap[swl[js+3]]] += 1
            else:
                ig[nodemap[swl[js]]][nodemap[swl[js+1]]] += 1
                ig[nodemap[swl[js]]][nodemap[swl[js+2]]] += 1
                ig[nodemap[swl[js]]][nodemap[swl[js+3]]] += 1
    if b[bi] == '0':
        if rlen>1:
            ng[nodemap[swl[slen]]][nodemap[swl[slen+1]]] += 1
        if rlen >2:
            ng[nodemap[swl[slen]]][nodemap[swl[slen+2]]] += 1
            ng[nodemap[swl[slen+1]]][nodemap[swl[slen+2]]] += 1
    else:
        if rlen>1:
            ig[nodemap[swl[slen]]][nodemap[swl[slen+1]]] += 1
        if rlen>2:
            ig[nodemap[swl[slen]]][nodemap[swl[slen+2]]] += 1
            ig[nodemap[swl[slen+1]]][nodemap[swl[slen+2]]] += 1
    bi += 1

# score sentences
i = 0
for s in sentences:
    gfs = []
    swl = s.split()
    dellist = []
    for w in swl:
        if w not in allowed:
            dellist.append(w)
    for dw in dellist:
        swl.remove(dw)
    slen = len(swl)-3
    rlen = len(swl)
    gsize = (3*slen)+3
    if gsize == 0:
        gfs.append(0.0)
        gfs.append(0.0) 
    else:
        igs = 0
        ngs = 0
        if b[i] == '0':
            if rlen>3:
                j = 0
                for j in range(slen):
                    ng[nodemap[swl[j]]][nodemap[swl[j+1]]] -= 1
                    ng[nodemap[swl[j]]][nodemap[swl[j+2]]] -= 1
                    ng[nodemap[swl[j]]][nodemap[swl[j+3]]] -= 1
            if rlen>1:
                ng[nodemap[swl[slen]]][nodemap[swl[slen+1]]] -= 1
            if rlen>2:
                ng[nodemap[swl[slen]]][nodemap[swl[slen+2]]] -= 1
                ng[nodemap[swl[slen+1]]][nodemap[swl[slen+2]]] -= 1

            if rlen>3:
                j = 0
                for j in range(slen):
                    if ng[nodemap[swl[j]]][nodemap[swl[j+1]]] > 0:
                        ngs += 1
                        print(swl[j],swl[j+1])
                    if ng[nodemap[swl[j]]][nodemap[swl[j+2]]] > 0:
                        ngs += 1 
                        print(swl[j],swl[j+2])
                    if ng[nodemap[swl[j]]][nodemap[swl[j+2]]] > 0:
                        ngs += 1 
                        print(swl[j],swl[j+3])
                    if ig[nodemap[swl[j]]][nodemap[swl[j+1]]] > 0:
                        igs += 1
                        print(swl[j],swl[j+1])
                    if ig[nodemap[swl[j]]][nodemap[swl[j+2]]] > 0:
                        igs += 1 
                        print(swl[j],swl[j+2])
                    if ig[nodemap[swl[j]]][nodemap[swl[j+2]]] > 0:
                        igs += 1 
                        print(swl[j],swl[j+3])
            if rlen>1:
                if ng[nodemap[swl[slen]]][nodemap[swl[slen+1]]] > 0:
                    ngs += 1 
                    print(swl[slen],swl[slen+1])
            if rlen>2:
                if ng[nodemap[swl[slen]]][nodemap[swl[slen+2]]] > 0:
                    ngs += 1 
                    print(swl[slen],swl[slen+2])
                if ng[nodemap[swl[slen+1]]][nodemap[swl[slen+2]]] > 0:
                    ngs += 1 
                    print(swl[slen+1],swl[slen+2])
            if rlen>1:
                if ig[nodemap[swl[slen]]][nodemap[swl[slen+1]]] > 0:
                    igs += 1 
                    print(swl[slen],swl[slen+1])
            if rlen>2:
                if ig[nodemap[swl[slen]]][nodemap[swl[slen+2]]] > 0:
                    igs += 1 
                    print(swl[slen],swl[slen+2])
                if ig[nodemap[swl[slen+1]]][nodemap[swl[slen+2]]] > 0:
                    igs += 1 
                    print(swl[slen+1],swl[slen+2])

            if slen>3:
                j = 0
                for j in range(slen):
                    ng[nodemap[swl[j]]][nodemap[swl[j+1]]] += 1
                    ng[nodemap[swl[j]]][nodemap[swl[j+2]]] += 1
                    ng[nodemap[swl[j]]][nodemap[swl[j+3]]] += 1
            if rlen>1:
                ng[nodemap[swl[slen]]][nodemap[swl[slen+1]]] += 1
            if rlen>2:
                ng[nodemap[swl[slen]]][nodemap[swl[slen+2]]] += 1
                ng[nodemap[swl[slen+1]]][nodemap[swl[slen+2]]] += 1
        else:
            if slen>3:
                j = 0
                for j in range(slen):
                    ig[nodemap[swl[j]]][nodemap[swl[j+1]]] -= 1
                    ig[nodemap[swl[j]]][nodemap[swl[j+2]]] -= 1
                    ig[nodemap[swl[j]]][nodemap[swl[j+3]]] -= 1
            if rlen>1:
                ig[nodemap[swl[slen]]][nodemap[swl[slen+1]]] -= 1
            if rlen>2:
                ig[nodemap[swl[slen]]][nodemap[swl[slen+2]]] -= 1
                ig[nodemap[swl[slen+1]]][nodemap[swl[slen+2]]] -= 1

            if slen>3:
                j = 0
                for j in range(slen):
                    if ig[nodemap[swl[j]]][nodemap[swl[j+1]]] > 0:
                        igs += 1
                        print(swl[j],swl[j+1])
                    if ig[nodemap[swl[j]]][nodemap[swl[j+2]]] > 0:
                        igs += 1 
                        print(swl[j],swl[j+2])
                    if ig[nodemap[swl[j]]][nodemap[swl[j+2]]] > 0:
                        igs += 1 
                        print(swl[j],swl[j+3])
                    if ng[nodemap[swl[j]]][nodemap[swl[j+1]]] > 0:
                        ngs += 1
                        print(swl[j],swl[j+1])
                    if ng[nodemap[swl[j]]][nodemap[swl[j+2]]] > 0:
                        ngs += 1 
                        print(swl[j],swl[j+2])
                    if ng[nodemap[swl[j]]][nodemap[swl[j+2]]] > 0:
                        ngs += 1 
                        print(swl[j],swl[j+3])
            if rlen>1:
                if ig[nodemap[swl[slen]]][nodemap[swl[slen+1]]] > 0:
                    igs += 1 
                    print(swl[slen],swl[slen+1])
            if rlen>2:
                if ig[nodemap[swl[slen]]][nodemap[swl[slen+2]]] > 0:
                    igs += 1 
                    print(swl[slen],swl[slen+2])
                if ig[nodemap[swl[slen+1]]][nodemap[swl[slen+2]]] > 0:
                    igs += 1 
                    print(swl[slen+1],swl[slen+2])
            if rlen>1:
                if ng[nodemap[swl[slen]]][nodemap[swl[slen+1]]] > 0:
                    ngs += 1 
                    print(swl[slen],swl[slen+1])
            if rlen>2:
                if ng[nodemap[swl[slen]]][nodemap[swl[slen+2]]] > 0:
                    ngs += 1 
                    print(swl[slen],swl[slen+2])
                if ng[nodemap[swl[slen+1]]][nodemap[swl[slen+2]]] > 0:
                    ngs += 1 
                    print(swl[slen+1],swl[slen+2])

            if rlen>3:
                j = 0
                for j in range(slen):
                    ig[nodemap[swl[j]]][nodemap[swl[j+1]]] += 1
                    ig[nodemap[swl[j]]][nodemap[swl[j+2]]] += 1
                    ig[nodemap[swl[j]]][nodemap[swl[j+3]]] += 1
            if rlen>1:
                ig[nodemap[swl[slen]]][nodemap[swl[slen+1]]] += 1
            if rlen>2:
                ig[nodemap[swl[slen]]][nodemap[swl[slen+2]]] += 1
                ig[nodemap[swl[slen+1]]][nodemap[swl[slen+2]]] += 1
        gfs.append(igs/gsize)
        gfs.append(ngs/gsize)
    graphscores.write(" ".join(map(str, gfs))+"\n")
    i += 1



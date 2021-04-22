#

# For citation, please check:
# https://github.com/teghub/IronyTR

# This code is a helper for feature extraction.
# You may need to create the missing files.


lookup = open("../lookup/lookup.txt", "r")
data = open("../data/lemmatized-whole.txt", "r")
binary = open("../data/binary-whole.txt", "r")
boosters = open("../lookup/boosters.txt", "r")
interjections = open("../lookup/interjections.txt", "r")
numbers = open("../lookup/numbers.txt", "r")
emotes = open("../lookup/emoji-emoticons.txt", "r")
bow = open("../data/features/bow-features-whole.txt", "r")

print("Helper for extracting normalized features. Some information:")
print("bow set: Bag-of-Words vectors as features")
print("basic set: Basic syntactic and lexical features on top of BoW vectors")
print("polarity set: Polarity score based features on top of basic set")
print("graph set: Graph containment similarity score based features on top of basic set")
print("polarity-graph: all features i.e. graph containment similarities on top of polarity set")
print("------")
print("To select a feature set document to create, type in the name of the feature set.")
print("feature set (bow/basic/polarity/graph/polarity-graph):")
file = input()
if file != "bow":
	print("If you do not want to add Bag-of-Words vectors to the set type \"n\". It is adviced to add BoW.")
	addbow = input()

if file == 'graph' or file == 'polarity-graph': 
	graphscores = open("../data/graph-scores.txt", "r")
	gs = [line.rstrip('\n') for line in graphscores]

if addbow == 'n':
	out = open("../data/features/no-bow-" + file + "-features-whole.txt", "a+")
else:
	out = open("../data/features/" + file + "-features-whole.txt", "a+")

sentences = [line.rstrip('\n') for line in data]
binaryval = [line.rstrip('\n') for line in binary]

punctmarks = ['?','!','...','(!)','(?)','"']
stopwords = ["bir",'*',':',',','.','/','(',')','-','REPEAT','CAPS']
boost = [line.rstrip('\n') for line in boosters]
interject = [line.rstrip('\n') for line in interjections]
nums = [line.rstrip('\n') for line in numbers]
emos = [line.rstrip('\n') for line in emotes]
bowvs = [line.rstrip('\n') for line in bow]

polarity = {}
lookuplist = [line.rstrip('\n') for line in lookup]

for l in lookuplist:
	linelist = l.split(" ")
	polarity[linelist[0]] = float(linelist[1])

i = 0
for s in sentences:
	tokencount = 0
	wordcount = 0
	word_token = 0
	pmcount = 0
	pmcount_token = 0
	em_exists = 0
	em_count = 0
	em_pmcount = 0
	qm_exists = 0
	qm_count = 0
	qm_pmcount = 0
	pem_exists = 0
	pem_count = 0
	pem_pmcount = 0
	pqm_exists = 0
	pqm_count = 0
	pqm_pmcount = 0
	quote_exists = 0
	quote_count = 0
	quote_pmcount = 0
	ellipsis_exists = 0
	ellipsis_count = 0
	ellipsis_pmcount = 0
	emo_exists = 0
	emo_count = 0
	emo_token = 0
	interject_exists = 0
	repet_exists = 0
	caps_exists = 0
	boost_exists = 0
	polarity_sum = 0
	polsum_token = 0
	pos_sum = 0
	pos_sum_token = 0
	neg_sum = 0
	neg_sum_token = 0
	max_pol = 0
	min_pol = 0
	max_min_diff = 0
	possum_negsum_diff = 0
	clash = 0
	if file == 'graph' or file == 'polarity-graph':
		i_graph = ((gs[i]).split(" "))[0]
		ni_graph = ((gs[i]).split(" "))[1]

	poslist = []
	neglist = []
	fs = []
	swl = s.split()
	tokencount = len(swl)
	wordcount = len(swl)
	
	for w in swl:
		if w in stopwords:
			if w == 'REPEAT':
				repet_exists = 1
			if w == 'CAPS':
				caps_exists = 1
			tokencount -= 1
			wordcount -= 1
		elif w in nums:
			tokencount -= 1
			wordcount -= 1
		elif w in punctmarks:
			pmcount += 1
			wordcount -= 1
			if w == '!':
				em_exists = 1
				em_count += 1
			elif w == '?':
				qm_exists = 1
				qm_count += 1
			elif w == '(!)':
				pem_exists = 1
				pem_count += 1
			elif w == '(?)':
				pqm_exists = 1
				pqm_count += 1
			elif w == '"':
				quote_exists = 1
				quote_count += 1
			else:
				ellipsis_exists = 1
				ellipsis_count += 1

		else:
			sc = polarity.get(w, 0)
			if sc != 0:
				polarity_sum += sc
				if sc > 0:
					poslist.append(sc)
					pos_sum += sc
				else:
					neglist.append(sc)
					neg_sum += sc
			if w in boost:
				boost_exists = 1
			if w in interject:
				interject_exists = 1
			if w in emos:
				emo_exists = 1
				emo_count += 1
				wordcount -= 1

	if len(poslist)>0 and len(neglist)>0:
		clash = 1

	if len(poslist)>0:
		pos_sum_token = pos_sum/(len(neglist)+len(poslist))
		polsum_token = ((polarity_sum/(len(neglist)+len(poslist)))+1)/2
		poslist.sort()
		max_pol = poslist[-1]

	if len(neglist)>0:
		neg_sum_token = -(neg_sum/(len(neglist)+len(poslist))) #normalization
		polsum_token = ((polarity_sum/(len(neglist)+len(poslist)))+1)/2
		neglist.sort()
		min_pol = -neglist[0] #normalization 

	max_min_diff = (max_pol+min_pol)/2 #normalization
	possum_negsum_diff = (pos_sum_token+neg_sum_token)/2 #normalization
	word_token = wordcount/tokencount
	emo_token = emo_count/tokencount
	pmcount_token = pmcount/tokencount
	
	if pmcount > 0:
		em_pmcount = em_count/pmcount
		qm_pmcount = qm_count/pmcount
		pem_pmcount = pem_count/pmcount
		pqm_pmcount = pqm_count/pmcount
		quote_pmcount = quote_count/pmcount
		ellipsis_pmcount = ellipsis_count/pmcount
	
	#fs.append(s)
	if addbow == 'n':
		fs.append(int(binaryval[i]))
	fs.append(word_token)
	fs.append(pmcount_token)
	fs.append(em_exists)
	fs.append(em_pmcount)
	fs.append(qm_exists)
	fs.append(qm_pmcount)
	fs.append(pem_exists)
	fs.append(pem_pmcount)
	fs.append(pqm_exists)
	fs.append(pqm_pmcount)
	fs.append(quote_exists)
	fs.append(quote_pmcount)
	fs.append(ellipsis_exists)
	fs.append(ellipsis_pmcount)
	fs.append(emo_exists)
	fs.append(emo_token)
	fs.append(interject_exists)
	fs.append(repet_exists)
	fs.append(boost_exists)
	fs.append(caps_exists)
	if file == 'polarity' or file == 'polarity-graph':
		fs.append(polsum_token)
		fs.append(pos_sum_token)
		fs.append(neg_sum_token)
		fs.append(max_pol)
		fs.append(min_pol)
		fs.append(max_min_diff)
		fs.append(possum_negsum_diff)
		fs.append(clash)
	if file == 'graph' or file == 'polarity-graph':
		fs.append(i_graph)
		fs.append(ni_graph)

	if addbow != 'n':
		out.write(bowvs[i] + ", ")
	out.write(", ".join(map(str, fs))+"\n")
	print(fs)
	i += 1
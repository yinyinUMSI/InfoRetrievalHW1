import nltk
nltk.download('punkt')
# sentence = "At eight o'clock on Thursday morning, Arthur didn't feel very good."
file = open('ehr.txt','r')
lines = file.readlines()

frequency={}
totalLen = 0
for line in lines:
	token_line = nltk.word_tokenize(line)
	for word in token_line:
		if ord(word[0].lower())<=122 and ord(word[0].lower())>=97:
			totalLen += 1
			if word not in frequency:
				frequency[word.lower()] = 0
			frequency[word.lower()] += 1
print(frequency)

stopwords = open('stoplist.txt', 'r')
stopwordsLst = stopwords.read().splitlines()


##a

stopwrdsTotal = 0
for i in list(frequency):
	if i in stopwordsLst:
		stopwrdsTotal += frequency[i]
answerA = stopwrdsTotal/totalLen
print('A:',answerA)

##b
characNumber = 0
capitalNumber = 0
for line in lines:
	token_line = nltk.word_tokenize(line)
	for word in token_line:
		if ord(word[0].lower())<=122 and ord(word[0].lower())>=97:
			characNumber += len(word)
			for i in word:
				if ord(i)<=90 and ord(i)>=65:
					capitalNumber+= 1
print('B:',capitalNumber/characNumber)

##c
characNumber = 0
accu = 0
for i in list(frequency):
	characNumber += len(i)*frequency[i]
	accu += frequency[i]
answerC = characNumber/accu
print('C:',answerC)


##d
nltk.download('averaged_perceptron_tagger')
noun = 0
verb = 0
adverb = 0
pronouns = 0
adj = 0
totalLen = 0
for line in lines:
	token_line = nltk.word_tokenize(line)
	for item in nltk.pos_tag(token_line):
		if ord(item[0][0].lower())<=122 and ord(item[0][0].lower())>=97:
			totalLen += 1
			if item[1][0:2] == 'NN':
				noun += 1
			if item[1][0:2] == 'VB':
				verb += 1
			if item[1][0:2] == 'RB':
				adverb += 1
			if item[1][0:2] == 'JJ':
				adj += 1
			if item[1][0:3] == 'PRP':
				pronouns += 1
print('D:',noun/totalLen, verb/totalLen, adverb/totalLen, pronouns/totalLen, adj/totalLen)

##e
word_type = []
types = ''
for line in lines:
	token_line = nltk.word_tokenize(line)
	for item in nltk.pos_tag(token_line):
		if ord(item[0][0].lower())<=122 and ord(item[0][0].lower())>=97:
			if item[0].lower() not in stopwordsLst:
				if item[1][0:2] == 'NN':
					types = 'noun'
				if item[1][0:2] == 'VB':
					types = 'verb'
				if item[1][0:2] == 'RB':
					types = 'adverb'
				if item[1][0:2] == 'JJ':
					types = 'adj'
				word_type.append([item[0], types])

import pandas as pd
import numpy as np
df = pd.DataFrame(word_type)
grouped = df.reset_index().groupby([0,1]).count().reset_index()
nounForm = grouped.loc[grouped[1] == 'noun'].sort_values(by='index',ascending=False).head(10)
print('e:',nounForm)

####TF_IDF

def calcu_TF(term, doc_frequency, docuLen):
	ctd = doc_frequency[term]/docuLen
	return np.log10(ctd+1)
def calcu_IDF(term):
	N = len(lines)
	k = 1
	for line in lines:
		if term.lower() in nltk.word_tokenize(line.lower()):
			k += 1
	return 1+np.log10(N/k)
for line in lines[0:10]:
	tokenized = nltk.word_tokenize(line)
	docuLen = len(tokenized)
	doc_frequency = {}
	for i in tokenized:
		if ord(i[0].lower())<=122 and ord(i[0].lower())>=97:
			if i not in doc_frequency:
				doc_frequency[i.lower()] = 0
			doc_frequency[i.lower()] += 1
	for item in doc_frequency:
		TFIDF = calcu_IDF(item)*calcu_TF(item, doc_frequency, docuLen)
		doc_frequency[item] = TFIDF
		# print(doc_frequency)
	calTFIDF = pd.Series(doc_frequency).to_frame('tfidf')
	sortedTFIDF = calTFIDF.sort_values(by='tfidf', ascending=False).head(10)
	print(sortedTFIDF)
file.close()

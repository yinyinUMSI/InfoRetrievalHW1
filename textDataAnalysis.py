import nltk
nltk.download('punkt')
# sentence = "At eight o'clock on Thursday morning, Arthur didn't feel very good."
file = open('ehr.txt','r')
lines = file.readlines()
file.close()
frequency={}
for line in lines:
	token_line = nltk.word_tokenize(line)
	for word in token_line:
		if ord(word[0].lower())<=122 and ord(word[0].lower())>=97:
			if word not in frequency:
				frequency[word.lower()] = 0
			frequency[word.lower()] += 1

stopwords = open('stoplist.txt', 'r')
stopwordsLst = stopwords.read().splitlines()
# print(stopwordsLst)

for i in list(frequency):
	if i in stopwordsLst:
		del frequency[i]

sortedFrequency = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
totalLength = len(frequency)
print(totalLength)
###start plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.Series(frequency).to_frame('times')
toPlot = df.reset_index().groupby(['times']).count().reset_index()
print(toPlot)
plt.plot(np.log10(toPlot['times']), np.log10(toPlot['index']/2745))

plt.savefig("easyplot2.jpg")

import readnarratives
import numpy
import re
from sklearn.feature_extraction.text import CountVectorizer


def bagOfWordsVector(text_data):
	vectorizer = CountVectorizer(min_df=1,stop_words='english')
	X = vectorizer.fit_transform(text_data)
	print (vectorizer.get_feature_names())
	return X

def getModelData(data):
	texts = [narrative[0] for narrative in data]
	y = numpy.array([narrative[1] for narrative in data])

	return texts,y

def processText(text_data):
	clean_text = []
	for text in text_data:
		clean_text.append(removeParentheses(text))
	return clean_text

def removeParentheses(text):
	no_parentheses = re.sub(r'\s?\([^)]*\)', '', text)
	return no_parentheses

def main():
	data = readnarratives.loadNarrativeData('agency', first = 49, last = 49)
	texts,y = getModelData(data)
	clean_texts = processText(texts)

	X = bagOfWordsVector(clean_texts)
	print (X)
	#print(removeParentheses('ajhflaskdjfh adjfhalskfj (adjfalskfjhas). adfhaljskdfh (ajsdhflkasdf)'))
if __name__ == '__main__':
	main()
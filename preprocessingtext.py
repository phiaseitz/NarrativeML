import readnarratives
import re
import nltk

def processText(text_data):
	clean_texts = []
	for text in text_data:
		no_parens = removeParentheses(text)
		only_alpha_spaces = removeNonAlphabet(no_parens)
		one_space = removeExtraSpaces(only_alpha_spaces)

		stemmed_text = stemWords(one_space)

		clean_texts.append(stemmed_text.lower())
	return clean_texts

def removeNonAlphabet(text):
	alpha = re.compile('[a-zA-Z ]*')
	only_alphabet_list = alpha.findall(text)
	only_alphabet = ''.join(only_alphabet_list)
	return only_alphabet

def removeParentheses(text):
	no_parentheses = re.sub(r'\s?\([^)]*\)', '', text)
	return no_parentheses

def removeExtraSpaces(text):
	one_space = re.sub(r'\s+',' ', text)
	return one_space

def stemWords(text):
	words = text.split(' ')
	stemmed_text = ''
	stemmer = nltk.stem.snowball.SnowballStemmer("english", ignore_stopwords=True)
	#stemmer = nltk.stem.porter.PorterStemmer()
	for word in words:
		stemmed_text = stemmed_text + ' ' + stemmer.stem(word)
	return stemmed_text

def discardBlanks (texts, scores):
	new_texts = []
	new_scores = []
	for i,text in enumerate(texts):
		if text != '':
			new_texts.append(text)
			new_scores.append(scores[i])
	return new_texts,new_scores


def processAndPickle(file_name, dimension = 'agency', first = 1, last = 140):
	data = readnarratives.loadNarrativeData(dimension, first, last)
	texts = [narrative[0] for narrative in data]
	scores = [narrative[1] for narrative in data]

	clean_texts = processText(texts)

	pickle_texts,pickle_scores = discardBlanks(clean_texts,scores)

	readnarratives.makePickle(pickle_texts,pickle_scores, file_name)

	text,score = readnarratives.readPickle(file_name)

def main():
	
	file_name = 'NarrativePickleAgency'

	processAndPickle(file_name, 'agency', 1, 140)

	text,score = readnarratives.readPickle(file_name)

	print text
	print len(text)
	print score
	print len(score)

if __name__ == '__main__':
	main()
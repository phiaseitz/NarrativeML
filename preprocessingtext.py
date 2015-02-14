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
		print (stemmed_text.lower())
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
	stemmer = nltk.stem.porter.PorterStemmer()
	for word in words:
		stemmed_text = stemmed_text + ' ' + stemmer.stem(word)
	return stemmed_text

def processAndPickle(file_name):
	data = readnarratives.loadNarrativeData('agency', first = 49, last = 49)
	texts = [narrative[0] for narrative in data]
	scores = [narrative[1] for narrative in data]

	clean_texts = processText(texts)

	readnarratives.makePickle(clean_texts,scores, file_name)

	text,score = readnarratives.readPickle(file_name)

def main():
	data = readnarratives.loadNarrativeData('agency', first = 49, last = 49)
	texts = [narrative[0] for narrative in data]
	scores = [narrative[1] for narrative in data]

	clean_texts = processText(texts)



	file_name = 'NarrativePickle'

	readnarratives.makePickle(clean_texts,scores, file_name)

	text,score = readnarratives.readPickle(file_name)


if __name__ == '__main__':
	main()
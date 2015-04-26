import readnarratives
import re
import nltk

def processText(text_data):
	"""Given a list of text data, return a list of text that has no parentheses
	 only alhpabet characters and spaces. Then, we stem the text using a porter
	 2 stemmer and append each cleaned text to a list that we return"""
	clean_texts = []
	for text in text_data:
		no_parens = removeParentheses(text)
		only_alpha_spaces = removeNonAlphabet(no_parens)
		one_space = removeExtraSpaces(only_alpha_spaces)

		stemmed_text = stemWords(one_space)

		clean_texts.append(stemmed_text.lower())
	return clean_texts

def removeNonAlphabet(text):
	"""Remove anything that's not an alphabet character or a space from text"""
	alpha = re.compile('[a-zA-Z .]*')
	only_alphabet_list = alpha.findall(text)
	only_alphabet = ''.join(only_alphabet_list)
	return only_alphabet

def removeParentheses(text):
	"""Remove anything that's in parentheses from text"""
	#print text
	no_parentheses = re.sub(r'\s?\([^)]*\)', '', text)
	return no_parentheses

def removeExtraSpaces(text):
	"""Remove any extra soaces from text"""
	one_space = re.sub(r'\s+',' ', text)
	return one_space

def stemWords(text):
	"""Stem words using the snowball stemmer (Porter 2)"""
	words = text.split(' ')
	stemmed_text = ''
	stemmer = nltk.stem.snowball.SnowballStemmer("english", 
		ignore_stopwords=True)
	#stemmer = nltk.stem.porter.PorterStemmer()
	for word in words:
		stemmed_text = stemmed_text + ' ' + stemmer.stem(word)
	return stemmed_text

def discardBlanks (texts, scores):
	"""Discard any black text entries and throw out shose scoress"""
	new_texts = []
	new_scores = []
	for i,text in enumerate(texts):
		if text != '' or text != ' ':
			new_texts.append(text)
			new_scores.append(scores[i])
	return new_texts,new_scores


def processAndPickle(file_name, dimension = 'agency', first = 1, last = 140):
	"""Go from reading the narratives to making a processed pickle of 
	all the data"""
	clean_responses = []
	responses = readnarratives.loadNarrativeData(dimension, first, last)
	print responses [0]

	#print responses [0][1]


	for narrative in responses:
		for scene in narrative:
			scene_text = scene[0][0]
			scene_score = scene[0][1]

			tagged_texts = [text[0] for text in scene[1]]
			tagged_scores = [text[1] for text in scene[1]]

			scene_clean_text = processText([scene_text])[0]
			tagged_clean_texts = processText(tagged_texts)

			no_blank_texts, no_blank_scores = discardBlanks(tagged_clean_texts,
				tagged_scores)

			clean_examples = [(text,no_blank_scores[i]) 
				for i,text in enumerate(no_blank_texts)]
			clean_responses.append(((scene_clean_text,scene_score),
				clean_examples))

	readnarratives.makePickle(clean_responses, file_name)

	data = readnarratives.readPickle(file_name)

def main():
	
	file_name = 'NarrativePickleAgency_test'

	processAndPickle(file_name, 'agency', 1, 16)

	data = readnarratives.readPickle(file_name)

	#print data
	
if __name__ == '__main__':
	main()
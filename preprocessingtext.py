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
	alpha = re.compile('[a-zA-Z .]*')
	only_alphabet_list = alpha.findall(text)
	only_alphabet = ''.join(only_alphabet_list)
	return only_alphabet

def removeParentheses(text):
	print text
	no_parentheses = re.sub(r'\s?\([^)]*\)', '', text)
	return no_parentheses

def removeExtraSpaces(text):
	one_space = re.sub(r'\s+',' ', text)
	return one_space

def stemWords(text):
	words = text.split(' ')
	stemmed_text = ''
	stemmer = nltk.stem.snowball.SnowballStemmer("english", 
		ignore_stopwords=True)
	#stemmer = nltk.stem.porter.PorterStemmer()
	for word in words:
		stemmed_text = stemmed_text + ' ' + stemmer.stem(word)
	return stemmed_text

def discardBlanks (texts, scores):
	new_texts = []
	new_scores = []
	for i,text in enumerate(texts):
		if text != '' or text != ' ':
			new_texts.append(text)
			new_scores.append(scores[i])
	return new_texts,new_scores


def processAndPickle(file_name, dimension = 'agency', first = 1, last = 140):
	responses = readnarratives.loadNarrativeData(dimension, first, last)
	print responses

	clean_responses = []


	for narrative in responses:
		scene_text = narrative[0][0]
		scene_score = narrative [0][1]

		tagged_texts = [text[0] for text in narrative[1]]
		tagged_scores = [text[1] for text in narrative[1]]

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

	print data
	
if __name__ == '__main__':
	main()
import difflib
import readnarratives
import unicodedata

def findOverlap (s1, s2):
	"""Find the number of words that overlap in s1 and s2"""
	s1_words = s1.split(' ')
	s2_list = s2.split(' ')
	s2_words = [word.replace('.','') for word in s2_list]

	s = difflib.SequenceMatcher(None, s1_words, s2_words)
	match = s.find_longest_match(0, len(s1_words), 0, len(s2_words))

	return match.size
	

def getScore(sentence, example_texts, example_scores):
	"""For each sentence, get the score for each sentence based on what
	examle the sentence is from"""
	max_overlap = 0
	max_overlap_i = -1

	# print sentence

	for i,example_text in enumerate(example_texts):
		# If a sentence is completely contained in an example
		if (sentence in example_text):
			# print example_scores[i]
			# print example_texts[i]
			# print ('\n')
			return example_scores[i]
		else:
		# calculate overlap with each string 
			overlap = findOverlap(sentence, example_text)
			if overlap > max_overlap:
				max_overlap = overlap
				max_overlap_i = i
	if max_overlap > 0:
		# print example_scores[i]
		# print example_texts[i]
		# print ('\n')
		return example_scores[max_overlap_i]


		#Otherwise, find the example with the most overlap.


def getSentenceScores (narrative_response):
	"""Given the response from a narrative return a list of sentences and their 
	scores"""

	scored_sentences = []
	scene_text = narrative_response [0][0]
	scene_score = narrative_response [0][1] 

	example_texts = [text[0] for text in narrative_response[1]]
	example_scores = [text[1] for text in narrative_response[1]]

	sentences_blanks = scene_text.split('.')
	sentences = [x for x in sentences_blanks if (x and x != u' ')]
	#print sentences

	# for i in range(len(example_scores)):
	# 	print (example_texts[i])
	# 	print (example_scores[i])

	for sentence in sentences:
		 scored_sentences.append((sentence,
		 	getScore(sentence, example_texts, example_scores)))

	return scored_sentences

def getSentenceData(data):
	"""This takes in an entire narrative's worth of data and scores is based 
	on each scene. """
	sentence_data = []
	for scene in data:
		sentence_data = sentence_data + getSentenceScores(scene)

	return sentence_data

def main():
	
	file_name = 'NarrativePickleAgency_test'

	data = readnarratives.readPickle(file_name)

	print getSentenceData(data)

	#print(data[0][1])
	
if __name__ == '__main__':
	main()
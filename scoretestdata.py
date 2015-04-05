import readnarratives

def findOverlap (s1, s2):
	""" returns the longest common substring from the beginning of sa and sb """
	def _iter():
		for a, b in zip(s1, s2):
			if a == b:
		    	yield a
			else:
		    	return
	return len(''.join(_iter()))

def getScore(sentence, example_texts, example_scores):
	max_overlap = 0
	max_overlap_i = -1

	for i,example_text in enumerate(example_texts):
		# If a sentence is completely contained in an example
		if (sentence in example_text):
			return example_scores[i]
		# calculate overlap with each string 
		else:
			overlap = findOverlap(sentence, example_text)
			if overlap > max_overlap:
				max_overlap = overlap
				max_overlap_i = i

	return example_scores[max_overlap_i]
		#Otherwise, find the example with the most overlap.


def getSentenceScores (narrative_response):
	scene_text = narrative_response [0][0]
	scene_score = narrative_response [0][1] 

	example_text = [text[0] for text in narrative_response[1]]
	example_scores = [text[1] for text in narrative_response[1]]

	sentences = scene_text.split('.')

	for sentence in sentences:
		getScore(sentence, example_text, example_scores)	



def main():
	
	file_name = 'NarrativePickleAgency_test'

	data = readnarratives.readPickle(file_name)

	getSentenceScores(data[0])
	
if __name__ == '__main__':
	main()
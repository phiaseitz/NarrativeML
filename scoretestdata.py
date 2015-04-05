import readnarratives

def findOverlap (s1, s2):
	s1_words = s1.split(' ')
	s2_words = s2.split(' ')

	max_start_j = 0
	max_end_j = 0

	start_j = 1
	end_j = 0

	matching = False

	for start_i in range(len(s1_words)):
		i = start_i
		for j in range(len(s2_words)):
			if (s1_words[i]== s2_words[j]) and matching:
				end_j = j
				if i < len(s1_words)-1:
					i = i+1
			elif (s1_words[i] == s2_words[j]) and not(matching):
				matching = True
				start_j = j
				end_j = j
				if i < len(s1_words)-1:
					i = i+1
			elif matching and not(s1_words[i] == s2_words[j]):
				matching = False
				if (end_j-start_j) > (max_end_j-max_start_j):
					max_start_j = start_j
					max_end_j =  end_j
					i = start_i
		start_j = 0
		end_j = 0
		matching = False
	return (max_end_j-(max_start_j-1))

def getScore(sentence, example_texts, example_scores):
	max_overlap = 0
	max_overlap_i = -1

	for i,example_text in enumerate(example_texts):
		# If a sentence is completely contained in an example
		if (sentence in example_text):
			return example_scores[i]
		# calculate overlap with each string 
		elif example_text in sentence:
			overlap = len(example_text.split(' '))
			if overlap > max_overlap:
				max_overlap = overlap
				max_overlap_i = i
		else:
			overlap = findOverlap(sentence, example_text)
			if overlap > max_overlap:
				max_overlap = overlap
				max_overlap_i = i
	if max_overlap > 0:
		return example_scores[max_overlap_i]
	else:
		return None

		#Otherwise, find the example with the most overlap.


def getSentenceScores (narrative_response):
	scene_text = narrative_response [0][0]
	scene_score = narrative_response [0][1] 

	example_text = [text[0] for text in narrative_response[1]]
	example_scores = [text[1] for text in narrative_response[1]]

	sentences = scene_text.split('.')

	print example_scores
	for sentence in sentences:
		print(getScore(sentence, example_text, example_scores))



def main():
	
	file_name = 'NarrativePickleAgency_test'

	data = readnarratives.readPickle(file_name)

	getSentenceScores(data[0])

	
if __name__ == '__main__':
	main()
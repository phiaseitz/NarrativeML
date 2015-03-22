from termcolor import colored, cprint
import numpy

#Shift the weights so that they're between 0 and 1 (so that it's easier to 
#	print)
def normalizeWeights(weights):
	min_weight = min(weights)
	weights_pos = numpy.subtract(weights,min_weight)
	max_weight = max(weights)
	return weights_pos/max_weight

def normalizeWeightsList (weights_list):
	#I think that actual weights are positive and negative, so zero is a 
	#good starting point. We'll see

	#print weights_list

	min_weight = 1
	max_weight = -1

	#Find min and max
	for weights in weights_list:
		if max(weights) > max_weight:
			max_weight = max(weights)
		if min(weights) < min_weight:
			min_weight = min(weights)
	#print min_weight
	#print max_weight

	#Shift so that everything is between 0 and 1
	normalized_weights_list = []

	for weights in weights_list:
		weights_pos = numpy.subtract(weights,min_weight)
		normalized_weights_list.append(weights_pos/max_weight)

	return normalized_weights_list


#For each any given weight, determine the color
def findHighlight(weight):
	highlights=['on_green','on_cyan','on_blue','on_magenta', 'on_red']
	weight_ranges = numpy.linspace(0,1,len(highlights)+1)
	weight_ranges = weight_ranges[1:]

	for index,highlight in enumerate(highlights):
		if weight <= weight_ranges[index]:
			return highlight
	return highlights[-1]


#Given words and weights, print out the highlighted text
def printColored(text,weights):
	weights =  normalizeWeights(weights)
	for index,word in enumerate(text):
		highlight = findHighlight(weights[index])
		cprint(word,'white',highlight,end = " ")

#For all the words in the classifier, create a dictionary of word to weights
def createWeightsDictionary(vectorizer,classifier):
	feature_names = numpy.asarray(vectorizer.get_feature_names())
	#print (feature_names)
	feature_weights_dict = {}
	vec_shape = classifier.coef_.shape

	#print(vec_shape)

	for i in range(vec_shape[0]):
		feature_name = feature_names[i]
		#print (feature_name)
		feature_weights_dict[feature_name.encode('ascii','ignore')] = \
		 classifier.coef_[i]
	#print feature_weights_dict
	return feature_weights_dict

#Now, compute the real weight for each word by multiplying by the 
#	corresponding value in the TFIDF vector
def getWeightsFromDict(text,scene_vector,vectorizer, weights_dict):
	weights = []
	word_to_index = dict(zip(vectorizer.get_feature_names(), 
		range(len(vectorizer.get_feature_names()))))
	for i in range(len(text)):
		try:
			#The word is in the dictionary
			#print scene_vector[i]
			weights.append(weights_dict[text[i]]*
				scene_vector[0,word_to_index[text[i]]])
			#weights.append(weights_dict[text[i]])
		except:
			#The word is not in the dictionary
			weights.append(0)
			#print([text[i]])

	#print 
	return numpy.array(weights)

#Combines the two above functions
def getWeights(scene_text, scene_vector,vectorizer,classifier):
	weights_dict = createWeightsDictionary(vectorizer, classifier)
	return getWeightsFromDict(scene_text,scene_vector,vectorizer, 
		weights_dict)

#Given text, vectors, vectorizer, and classifier, print in colors!
def visualizeWeights(scene_text, scene_vector,vectorizer,classifier):
	split_list = scene_text.split(' ')
	split_list_ascii = [word.encode('ascii','ignore') for word in split_list]
	clean_split_list_ascii = [word for word in split_list_ascii if word != '']

	weights = getWeights(clean_split_list_ascii,scene_vector,vectorizer,
		classifier)
	printColored(clean_split_list_ascii,weights)

def splitText(scene_text):
	split_list = scene_text.split(' ')
	split_list_ascii = [word.encode('ascii','ignore') for word in split_list]
	clean_split_list_ascii = [word for word in split_list_ascii if word != '']

	return clean_split_list_ascii

def visualizeWeightsList(to_print, texts, vectors, scores, 
	vectorizer, classifier):
	classifier_weights_dict = createWeightsDictionary(vectorizer,classifier)

	scenes_weights = []
	scene_word_list = []

	for i in range(len(texts)):
		#append the weights for a scene
		scene_word_list.append(splitText(texts[i]))
		scenes_weights.append(getWeightsFromDict(scene_word_list[-1],
			vectors[i,:], vectorizer,classifier_weights_dict))

	normalized_weights = normalizeWeightsList(scenes_weights)

	for i in to_print:
		printColored(scene_word_list[i],normalized_weights[i])
		print ('\n Score: %i' %scores[i])

def printKey ():
	highlights=['on_green','on_cyan','on_blue','on_magenta', 'on_red']
	values = ['lowest', 'low', 'middle', 'high', 'highest']
	print ('Key :')

	for i in range(len(highlights)):
		cprint(values[i],'white',highlights[i])



def main():
	words = ['hi','hi','hi','hi','hi','hi','hi','hi','hi','hi','hi','hi']
	weights = numpy.linspace(0,10,len(words))

	printColored(words,weights)

	# 

	# print weight_ranges
	# print ('highlights')
	# print ('hello')

	# for highlight in highlights:
	# 	cprint('Can you read this?','white', highlight,end = " ")

if __name__ == '__main__':
	main()
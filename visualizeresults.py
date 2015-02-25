from termcolor import colored, cprint
import numpy

def normalizeWeights(weights):
	min_weight = min(weights)
	weights_pos = numpy.subtract(weights,min_weight)
	max_weight = max(weights)
	return weights_pos/max_weight


def findHighlight(weight):
	highlights=['on_blue','on_cyan','on_green','on_yellow','on_magenta','on_red']
	weight_ranges = numpy.linspace(0,1,len(highlights)+1)
	weight_ranges = weight_ranges[1:]

	for index,highlight in enumerate(highlights):
		if weight <= weight_ranges[index]:
			return highlight
	return highlights[-1]

def printColored(text,weights):
	weights =  normalizeWeights(weights)
	for index,word in enumerate(text):
		highlight = findHighlight(weights[index])
		cprint(word,'white',highlight,end = " ")

def createWeightsDictionary(scene_text, scene_vector,vectorizer,classifier):
	feature_names = numpy.asarray(vectorizer.get_feature_names())
	#print (feature_names)
	feature_weights_dict = {}
	vec_shape = scene_vector.shape

	for i in range(vec_shape[1]):
		feature_name = feature_names[i]
		print (feature_name)
		feature_weights_dict[feature_name.encode('ascii','ignore')] = classifier.coef_[i]
	print feature_weights_dict
	return feature_weights_dict

def getWeightsFromDict(text,weights_dict):
	weights = []
	for word in text:
		weights.append(weights_dict[word])


	return numpy.array(weights)

def getWeights(scene_text, scene_vector,vectorizer,classifier):
	weights_dict = createWeightsDictionary(scene_text, scene_vector,vectorizer,classifier)
	return getWeightsFromDict(scene_text,weights_dict)

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
import string
import os 
import pickle
import copy
from lxml import etree


def makePickle(responses,file_name):
	"""Make a pickle of the resonposes. Takes in a list of responses (including
	scores) and pickles them"""
	folder_path = '/Volumes/Research/Adler Research/Sophia OSS Stuff/'
	file_path = folder_path+file_name

	f = open(file_path, 'w')
	pickle.dump(responses, f)
	f.close()

def readPickle(file_name):
	"""Read and return whatever is in the pickle"""
	folder_path = '/Volumes/Research/Adler Research/Sophia OSS Stuff/'
	file_path = folder_path+file_name

	f = open(file_path, 'r')
	data = pickle.load(f)
	f.close()

	return data

def readNarrativeFile(narrative_number):
	"""Read an xml file of a narrative and return the xml object"""
	folder_path = '/Volumes/Research/Adler Research/Sophia OSS Stuff/' \
		'Narratives_xml_tagged/' 
	file_name = 'FLSA_{narrative_num}.xml'.format(narrative_num =
		str(narrative_number).zfill(3))
	file_path = folder_path+file_name
	
	if os.path.isfile(file_path):
		print ('reading narrative {narrative_num}'.format(
			narrative_num = str(narrative_number)))
		#Open the filse and read all the lines
		narrative_xml = etree.parse(file_path)
		return narrative_xml
	#If there's not a file at the path, then either that number
	#was skipped or I'm not connected to fsvs01
	else:
		if os.path.isdir(folder_path):
			print ('Narrative {narrative_name} does not exist'.
				format(narrative_name = file_name))
		else:
			print ('Are you connected to fsvs01 and have you' \
			 'mounted the research drive?')
		return None

	#print(etree.tostring(narrative_xml, pretty_print=True))

def getResponses(narrative_xml,coding_dimension):
	""" go from the xml object to  a list of responses. The list of the 
	responses (with scores) is a list of 
	((scene text, scene score),[(ex text, ex score)]) entries"""
	#Go from XML to list of (text,score) tuples
	responses = []
	scene_text = ''
	tagged = []
	whole_scene = []
	root = narrative_xml.getroot()
	root_attributes = root.attrib
	score_key = 'scene_'+coding_dimension
	for scene in root:
		#Get score for each scene (that's the overall score that we care about)
		scene_attributes = scene.attrib
		scene_score = scene_attributes[score_key]
		scene_responses = ''
		for passage in scene:
			passage_attributes = passage.attrib
			if passage_attributes['speaker'] == 'Respondent':
				for example in passage:
					#Get the example score
					example_attributes = example.attrib
					tagged.append((example.text,
						float(example_attributes['score'])))
					#Add the text from the example to the scene text
					scene_responses = scene_responses + ' ' + example.text
		#Don't forget to add the scene too
		responses.append(((scene_responses,float(scene_score)),
			copy.deepcopy(tagged)))
		del tagged[:]

	return responses

def loadNarrativeData(coding_dimension, first = 1, last = 164):
	"""Loads the narratives from first to last, and also pulls out the scores
	corresponding to the inputted coding dimension. returns the list of responses
	for each narrative in the format
	[((scene test, scene score),[(ex text, ex score)])]
	"""
	responses = []

	#Loop through all the narratives
	for narrative_number in range(first,last + 1):
		#Read the narrative
		narrative_xml = readNarrativeFile(narrative_number)
		if not(narrative_xml is None):
			narrative_responses = getResponses(narrative_xml,
				coding_dimension)
			responses.append(narrative_responses)

	return responses

def main():
	# #Narratives to start and end at
	# first_narrative = 1
	# last_narrative = 2

	# #Loop through all the narratives
	# for narrative_number in range(first_narrative,last_narrative + 1):
	# 	print ('reading narrative {narrative_num}'.format(
	# 		narrative_num = str(narrative_number)))
	# 	#Read the narrative
	# 	narrative_xml = readNarrativeFile(narrative_number)
	# 	responses_xml = getOnlyResponses(narrative_xml)
	data = loadNarrativeData('agency', 1, 1)
	print(data)

if __name__ == '__main__':
	main()
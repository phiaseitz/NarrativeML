import string
import os 
import pickle
import re
from lxml import etree


def makePickle(text, scores,file_name):
	folder_path = '/Volumes/Research/Adler Research/Sophia OSS Stuff/'
	file_path = folder_path+file_name

	data = (text,scores)

	f = open(file_path, 'w')
	pickle.dump(data, f)
	f.close()

def readPickle(file_name):
	folder_path = '/Volumes/Research/Adler Research/Sophia OSS Stuff/'
	file_path = folder_path+file_name

	f = open(file_path, 'r')
	data = pickle.load(f)
	f.close()

	texts = data[0]
	scores = data[1]

	return texts,scores

def readNarrativeFile(narrative_number):
	folder_path = '/Volumes/Research/Adler Research/Sophia OSS Stuff/' \
		'Narratives_xml/' 
	file_name = 'FLSA_{narrative_num}.xml'.format(narrative_num =
		str(narrative_number).zfill(3))
	file_path = folder_path+file_name
	
	if os.path.isfile(file_path):
		print ('reading narrative {narrative_num}'.format(
			narrative_num = str(narrative_number)))
		#Open the filse and read all the lines
		narrative_xml = etree.parse(file_path)
		#print(etree.tostring(narrative_xml, pretty_print=True))
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

def responsesList(narrative_xml,coding_dimension):
	root = narrative_xml.getroot()
	root_attributes = root.attrib
	responses_xml = etree.Element("narrative", 
		narrative_num = root_attributes['narrative_num'])
	scenes = []
	examples =[]
	for scene in root:
		scene_attributes = scene.attrib 
		scene_score = scene_attributes['scene_' + coding_dimension]
		scene_responses = ''
		for passage in scene:
			passage_attributes = passage.attrib
			if passage_attributes['speaker'] == 'Respondent':
				passage_string = etree.tostring(passage,pretty_print = True)
				passage_text = re.sub(r'</*passage[^>]*>','',passage_string)
				
				examples_strings = re.findall(r'<example .* </example>',
					passage_text)

				for example in examples_strings:
					example_score = re.search(r'"\d"',example)
					example_score = example_score.group()
					example_score_int = int(example_score[1])

					example_text = re.sub(r'</*example[^>]*>','',example)
					
					#print example_text
					#print example_score
					examples.append((example_text,example_score_int))
				passage_text = re.sub(r'</*example[^>]*>','',passage_text)
				scene_responses = scene_responses + ' ' + passage_text
		scenes.append((scene_responses,int(scene_score)))			

	#print(etree.tostring(responses_xml, pretty_print=True))
	#print examples
	#print scenes
	return scenes+examples

def loadNarrativeData(coding_dimension, first = 1, last = 164):
	all_responses = []
	for narrative_number in range(first,last + 1):
		print narrative_number
		#Read the narrative
		narrative_xml = readNarrativeFile(narrative_number)
		if not(narrative_xml is None):
			responses = responsesList(narrative_xml,coding_dimension)
			all_responses = all_responses + responses
	return all_responses

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
	data = loadNarrativeData('agency',132,132)
	#print(data)

if __name__ == '__main__':
	main()
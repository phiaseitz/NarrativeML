import string
from lxml import etree

def readNarrativeFile(narrative_number):
	folder_path = '/Volumes/Research/Adler Research/Sophia OSS Stuff/' \
		'Narratives_xml/' 
	file_name = 'FLSA_{narrative_num}.xml'.format(narrative_num =
		str(narrative_number).zfill(3))
	file_path = folder_path+file_name

	narrative_xml = etree.parse(file_path)
	
	#print(etree.tostring(narrative_xml, pretty_print=True))

	return narrative_xml

def getOnlyResponses(narrative_xml):
	root = narrative_xml.getroot()
	root_attributes = root.attrib
	responses_xml = etree.Element("narrative", 
		narrative_num = root_attributes['narrative_num'])
	for scene in root:
		scene_attributes = scene.attrib
		responses_xml.append(etree.Element('scene', 
				scene_name = scene_attributes['scene_name'], 
				scene_agency = scene_attributes['scene_agency'],
				scene_communion = scene_attributes['scene_agency']))
		scene_responses = ''
		for passage in scene:
			passage_attributes = passage.attrib
			if passage_attributes['speaker'] == 'Respondent':
				scene_responses = scene_responses + ' ' + passage.text
		current_scene = responses_xml[-1]
		current_scene.text = scene_responses

	print(etree.tostring(responses_xml, pretty_print=True))
	return responses_xml

def getAllResponses(first_narrative, last_narrative):
	responses = []
	#Loop through all the narratives
	for narrative_number in range(first_narrative,last_narrative + 1):
		#Read the narrative
		narrative_xml = readNarrativeFile(narrative_number)
		responses_xml = getOnlyResponses(narrative_xml)
		responses.append(responses_xml)
	return responses
def xmlToList(xml_response, coding_dimension):
	#root = xml_response.getroot()
	score_key = 'scene_'+coding_dimension
	narrative_responses = []
	for scene in xml_response:
		scene_attributes = scene.attrib
		narrative_responses.append((scene.text,
			int(scene_attributes[score_key])))
	return narrative_responses

def LoadNarrativeData(coding_dimension, first = 1, last = 164):
	all_responses_xml = getAllResponses(first,last)
	data = []
	for response in all_responses_xml:
		response_data = xmlToList(response, coding_dimension)
		data = data + response_data
	return data

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
	print(LoadNarrativeData('communion'))

if __name__ == '__main__':
	main()
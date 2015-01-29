import os
import re
import string
from lxml import etree


def readFile (num_narrative):
	#Where the folder is
	folder_path = '/Volumes/Research/Adler Research/Sophia OSS Stuff/Narratives_txt/'
	#Create the file name for each narratve and get the full path
	file_name = 'FLSA_{narrative_num}.txt'.format(narrative_num = str(num_narrative).zfill(3))
	file_path = folder_path + file_name
	if os.path.isfile(file_path):
		#Open the file and read all the lines
		narrative_file = open(file_path, 'r')
		narrative_text_lines = narrative_file.readlines()
		return narrative_text_lines
	else:
		if os.path.isdir(folder_path):
			print ('Narrative {narrative_name} does not exist'.format(narrative_name = file_name))
		else:
			print ('Are you connected to fsvs01 and have you mounted the research drive?')
		return []

def cleanText (narrative_lines):
	newline_characters = ['\n', '\r', '\r\n']

	no_blank_lines = [line for line in narrative_lines if not (line in newline_characters)]

	clean_lines = []
	for line in no_blank_lines:
		clean_line = re.sub(r'\s*\n', '',line).decode('utf-8')

		clean_lines.append(clean_line)
	return clean_lines

def removeSpeakerFromLine (line):
	clean_line = re.sub(r'\A\w*\s*\w*:\s*', '',line)
	return clean_line

def addTextToCurrentScene(narrative, passage_text, current_speaker):
	current_scene = narrative[-1]
	current_scene.append(etree.Element('passage', speaker = current_speaker))
	current_scene[-1].text = passage_text

def linesToXML (narrative_lines, narrative_number):
	#All the scenes
	scenes = ['HIGH POINT', 'LOW POINT', 'TURNING POINT']

	#Possible interviewer and participant tags in the txt files
	speakers_dict = {'I' : 'Interviewer', 'THE INTERVIEWER' : 'Interviewer', 'INTERVIEWER' : 'Interviewer','R' : 'Respondent', 'RESPONDENT' : 'Respondent' ,'MALE SPEAKER' : 'Respondent', 'FEMALE SPEAKER' : 'Respondent', 'INTERVIEWEE' : 'Respondent', 'THE INTERVIEWEE' : 'Respondent', 'PARTICIPANT' : 'Respondent', 'ANSWER' : 'Respondent'}
	
	first_scene = True
	current_scene = ''
	current_speaker = 'Interviewer'
	current_passage = ''

	narrative = etree.Element('narrative', narrative_num = str(narrative_number))

	for line in narrative_lines:

		#Find the index of the first colon - this is how we know who is speaking
		colon_index = line.find(':')

		#If this is a scene heading
		if line.upper() in scenes: 
			if first_scene:
				first_scene = False
			else:
				addTextToCurrentScene(narrative, current_passage, current_speaker)
				current_passage = ''
				current_speaker  = 'Interviewer'
			current_scene = line
			#print (current_scene)
			narrative.append(etree.Element('scene', scene_name = current_scene.title()))
		#if there's no colon then we dont know who is speaking
		elif colon_index == -1 or not(line[:colon_index] in speakers_dict):
			#print ('speaker remains the same: {speaker}'.format(speaker = current_speaker))
			#print (line)
			current_passage = current_passage + ' ' + line
		#If there's a colon early on -- we know we have a speaker
		elif line[:colon_index] in speakers_dict:
			passage_speaker = speakers_dict[line[:colon_index]]
			if passage_speaker == current_speaker:
				current_passage = current_passage + ' ' + line
			else:
				if current_scene:
					addTextToCurrentScene(narrative, current_passage, current_speaker)
					#print current_speaker
					#print current_passage
					current_passage = removeSpeakerFromLine(line)
					current_speaker = passage_speaker
	
			#print (current_speaker)
			#print (line)
	addTextToCurrentScene(narrative, current_passage, current_speaker)
	
	print(etree.tostring(narrative, pretty_print=True))		

def main():
	#Narratives to start and end at
	first_narrative = 1
	last_narrative = 10

		#Loop through all the narratives
	for narrative_number in range(first_narrative,last_narrative + 1):
		print ('reading narrative {narrative_num}'.format(narrative_num = str(narrative_number)))

		narrative_text = readFile(narrative_number)
		if narrative_text:
			#Get rid of the line that has the narrative number in it
			clean_text = cleanText(narrative_text[1:])
			#print (clean_text)
			linesToXML(clean_text, narrative_number)


if __name__ == '__main__':
	main()
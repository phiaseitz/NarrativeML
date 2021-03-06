import os
import re
import string
import csv
from lxml import etree

#Reading the narrative file by line
def readFile (num_narrative):
	"""Given the narrative number, return the list of the text in each line 
	of the .txt file """
	#Where the folder is
	folder_path = '/Volumes/Research/Adler Research/Sophia OSS Stuff/' \
		'Narratives_txt/' 
	#Create the file name for each narratve and get the full path
	file_name = 'FLSA_{narrative_num}.txt'.format(narrative_num =
		str(num_narrative).zfill(3))
	file_path = folder_path + file_name
	#If this file is actually a file (the narratives sometimes skip numbers)
	if os.path.isfile(file_path):
		#Open the file and read all the lines
		narrative_file = open(file_path, 'r')
		narrative_text_lines = narrative_file.readlines()
		return narrative_text_lines
	#If there's not a file at the path, then either that number
	#was skipped or I'm not connected to fsvs01
	else:
		if os.path.isdir(folder_path):
			print ('Narrative {narrative_name} does not exist'.
				format(narrative_name = file_name))
		else:
			print ('Are you connected to fsvs01 and have you' \
			 'mounted the research drive?')
		return []

#There are lots of empty lines in the narratives, plus we don't want to have 
#newlines in the existing lines
def cleanText (narrative_lines):
	"""Given the list of lines, remove all newline characters as well as any 
	blank lines"""
	newline_characters = ['\n', '\r', '\r\n']
	#Get rid of all the empty lines
	no_blank_lines = [line for line in narrative_lines if not (
		line in newline_characters)]

	clean_lines = []
	#Get rid of new lines at the end of lines that do have text
	for line in no_blank_lines:
		clean_line = re.sub(r'\s*\n', '',line).decode('utf-8')
		clean_lines.append(clean_line)
	return clean_lines

#After we know who the speaker is, get rid of that so that we only have
#what they said
def removeSpeakerFromLine (line):
	"""Remove the indicated speaker from the text of a line"""
	#the regular expression is: at the beginning of the string,
	#some letters, maybe a space, and then maybe some more letters followed
	#by a colon and then maybe some spaces
	clean_line = re.sub(r'\A\w*\s*\w*:\s*', '',line)
	return clean_line

#After a speaker is done talking, we want to add that passage 
#to our xml document
def addTextToCurrentScene(narrative, passage_text, current_speaker):
	"""Given the xml object, passage text, and the curent speaker, add the
	passage to the xml object in the current scene and tag it with the 
	current speaker"""
	#Get the most recent element in our xml file
	current_scene = narrative[-1]
	#Make the passage, set the speaker, then set the text
	current_scene.append(etree.Element('passage', speaker = current_speaker))
	current_scene[-1].text = passage_text

#Go from a list of lines to an xml object
def linesToXML (narrative_lines, narrative_number, narrative_scores):
	"""Given a list of lines, the number of the narrative, and list of scores, 
	go from the lines to an xml object"""
	#All the scenes
	scenes = ['HIGH POINT', 'LOW POINT', 'TURNING POINT']

	#Possible interviewer and participant tags in the txt files
	speakers_dict = {'I' : 'Interviewer', 'THE INTERVIEWER' : 'Interviewer', \
		'INTERVIEWER' : 'Interviewer','R' : 'Respondent', \
		'RESPONDENT' : 'Respondent' ,'MALE SPEAKER' : 'Respondent', \
		'FEMALE SPEAKER' : 'Respondent', 'INTERVIEWEE' : 'Respondent', \
		'THE INTERVIEWEE' : 'Respondent', 'PARTICIPANT' : 'Respondent', \
		'ANSWER' : 'Respondent'}
	
	scenes_index_dict = ({'HIGH POINT': (0,1), 'LOW POINT':(2,3) , 
		'TURNING POINT': (4,5)})
	#Initialize all the tracking variables
	first_scene = True
	current_scene = ''
	current_speaker = 'Interviewer'
	current_passage = ''

	#Make the narrative
	narrative = etree.Element('narrative', narrative_num = str(
		narrative_number))

	#Iterate through all the lines
	for line in narrative_lines:

		#Find the index of the first colon 
		#this is how we know who is speaking
		colon_index = line.find(':')

		#If this is a scene heading
		if line.upper() in scenes: 
			#If it's the first scene set the flag to false
			if first_scene:
				first_scene = False
			#If it's not the first scene, then we have to add 
			#the last speaker's passage before moving on to the next scene
			else:
				addTextToCurrentScene(narrative, current_passage, 
					current_speaker)
				#Reset tracking variables
				current_passage = ''
				current_speaker  = 'Interviewer'
			#Set the current scene
			current_scene = line
			agency_index = scenes_index_dict[current_scene.upper()][0]
			communion_index = scenes_index_dict[current_scene.upper()][1]

			#print(agency_index)
			#print(narrative_scores)
			#Add the next scene to our xml tree
			narrative.append(etree.Element('scene', 
				scene_name = current_scene.title(), 
				scene_agency = narrative_scores[agency_index],
				scene_communion = narrative_scores[communion_index]))
		#if there's no colon then we dont know who is speaking
		elif colon_index == -1 or not(line[:colon_index] in speakers_dict):
			current_passage = current_passage + ' ' + line
		#If there the part of the line before the colon is a speaker tag, 
		#then we know we have a speaker
		elif line[:colon_index] in speakers_dict:
			#Find out who is speaking
			passage_speaker = speakers_dict[line[:colon_index]]
			#If it's still the same person keep adding on to the 
			#current passage
			if passage_speaker == current_speaker:
				current_passage = (current_passage + ' ' + 
					removeSpeakerFromLine(line))
			#Otherwise, add the text to the current scene, and then reset the 
			#current passage and update the current speaker
			else:
				if current_scene:
					addTextToCurrentScene(narrative, current_passage, 
						current_speaker)

					current_passage = removeSpeakerFromLine(line)
					current_speaker = passage_speaker

	#Add the last passage to the narrative
	addTextToCurrentScene(narrative, current_passage, current_speaker)
	
	#Prit it so that it looks pretty
	print(etree.tostring(narrative, pretty_print=True))		
	return narrative

def saveXML (narrative_xml,narrative_number):
	"""Save the xml object as an xml file with the number in the name"""
	folder_path = '/Volumes/Research/Adler Research/Sophia OSS Stuff/' \
		'Narratives_xml/' 
	file_name = 'FLSA_{narrative_num}.xml'.format(narrative_num =
		str(narrative_number).zfill(3))
	full_file_path = folder_path+file_name

	e_tree_narrative = etree.ElementTree(narrative_xml)

	e_tree_narrative.write(full_file_path, pretty_print=True)

def makeScoresDict():
	""" Make a dictionary of narrative number to scores that we can use later"""
	scores = {}
	ha_ind = 1
	hc_ind = 2
	la_ind = 6
	lc_ind = 7
	ta_ind = 11
	tc_ind = 12

	file_path = '/Volumes/Research/Adler Research/Sophia OSS Stuff/' \
		'NarrativeScores.csv'
	f = csv.reader(open(file_path, "rU"), dialect=csv.excel_tab)
	
	data = [row for row in f]
	data = data[1:]

	for row in data:
		row_str = row[0]
		row_scores=row_str.split(',')
		#print (row_scores)
		num_narrative = row_scores[0]
		file_name = str(num_narrative)
		scores[file_name] = (row_scores[ha_ind], row_scores[hc_ind],
			row_scores[la_ind], row_scores[lc_ind], row_scores[ta_ind],
			row_scores[tc_ind])

	#print (scores)
	return scores


def main():
	#Narratives to start and end at
	first_narrative = 49
	last_narrative = 49

	scores = makeScoresDict()

	#Loop through all the narratives
	for narrative_number in range(first_narrative,last_narrative + 1):
		print ('reading narrative {narrative_num}'.format(
			narrative_num = str(narrative_number)))
		#Read the narrative
		narrative_text = readFile(narrative_number)
		if narrative_text:
			#Get rid of the line that has the narrative number in it and
			#clean the text
			clean_text = cleanText(narrative_text[1:])
			#go from list of lines to xml
			narrative_scores = scores[str(narrative_number)]
			narrative = linesToXML(clean_text, narrative_number,
				narrative_scores)
			#saveXML(narrative,narrative_number)



if __name__ == '__main__':
	main()
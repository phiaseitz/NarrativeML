import os
import re
import lxml


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

def getRidOfEmptyLines (narrative_lines):
	newline_characters = ['\n', '\r', '\r\n']
	return [line for line in narrative_lines if not (line in newline_characters)]

def linesToXML (narrative_lines):
	#All the scenes
	scenes = ['HIGH POINT', 'LOW POINT', 'TURNING POINT']

	#Possible interviewier and participant tags in the txt files
	interviewer = ['I', 'THE INTERVIEWER', 'INTERVIEWER']
	participant = ['R', 'RESPONDENT' ,'MALE SPEAKER', 'FEMALE SPEAKER', 'INTERVIEWEE', 'THE INTERVIEWEE', 'PARTICIPANT', 'ANSWER']
	
	current_scene = ''
	current_speaker = 'I'

	narrative_lines = getRidOfEmptyLines(narrative_lines)

	for line in narrative_lines:
		#Get rid of new lines and carriage returns
		line = line.rstrip()

		#Find the index of the first colon - this is how we know who is speaking
		colon_index = line.find(':')

		#If this is a scene heading
		if line.upper() in scenes:
			current_scene = line
			print (current_scene)
		#if there's no colon then we dont know who is speaking
		elif colon_index == -1:
			print ('speaker remains the same: {speaker}'.format(speaker = current_speaker))
			print (line)
		#If there's a colon early on -- we know we have a speaker
		elif colon_index < 30:
			current_speaker = line[:colon_index]
			print (current_speaker)
			print (line)


def main():
	#Narratives to start and end at
	first_narrative = 1
	last_narrative = 1

		#Loop through all the narratives
	for narrative_number in range(first_narrative,last_narrative + 1):
		print ('reading narrative {narrative_num}'.format(narrative_num = str(narrative_number)))

		narrative_text = readFile(narrative_number)

		if narrative_text:
			linesToXML(narrative_text)


if __name__ == '__main__':
	main()
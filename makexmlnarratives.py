import os
import re

#All the scenes
scenes = ['HIGH POINT', 'LOW POINT', 'TURNING POINT']

#Possible interviewier and participant tags in the txt files
interviewer = ['I', 'THE INTERVIEWER', 'INTERVIEWER']
participant = ['R', 'RESPONDENT' ,'MALE SPEAKER', 'FEMALE SPEAKER', 'INTERVIEWEE', 'THE INTERVIEWEE', 'PARTICIPANT', 'ANSWER']

#Narratives to start and end at
first_narrative = 1
last_narrative = 165

#Where the folder is
folder_path = '/Volumes/Research/Adler Research/Sophia OSS Stuff/Narratives_txt/'

#Loop through all the narratives
for i in range(first_narrative,last_narrative):
	print ('reading narrative {narrative_num}'.format(narrative_num = str(i)))
	#Create the file name for each narratve and get the full path
	file_name = 'FLSA_{narrative_num}.txt'.format(narrative_num = str(i).zfill(3))
	full_file_path = folder_path + file_name

	#if there is a file there
	if os.path.isfile(full_file_path):

		#Open the file and read all the lines
		narrative_file = open(full_file_path, 'r')
		narrative_text_lines = narrative_file.readlines()
		
		#print (narrative_text_lines)
		#Loop through lines
		for line in narrative_text_lines:
			#Get rid of new lines and carriage returns
			line = line.rstrip()

			#Find the index of the first colon - this is how we know who is speaking
			colon_index = line.find(':')

			#If this is a scene heading
			if line.upper() in scenes:
				print ('scene is ' + line)
			#if there's no colon then we dont know who is speaking
			elif colon_index == -1:
				print ('speaker is not indicated')
			#If there's a colon early on -- we know we have a speaker
			elif colon_index < 30:
				speaker = line[:colon_index]
				if not(speaker in interviewer) and not(speaker in participant):
					print (speaker)
	else:
		if os.path.isdir(folder_path):
			print ('Narrative {narrative_num} does not exist'.format(narrative_num = str(i)))
		else:
			print ('Are you connected to fsvs01 and have you mounted the research drive?')
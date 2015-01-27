import os
import re
import lxml


def readFile (folder_path, num_narrative):
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


def linesToSpeaker (narrative_lines, scenes, interviewer, participant):
	for line in narrative_lines:
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


def main():
	#All the scenes
	scenes = ['HIGH POINT', 'LOW POINT', 'TURNING POINT']

	#Possible interviewier and participant tags in the txt files
	interviewer = ['I', 'THE INTERVIEWER', 'INTERVIEWER']
	participant = ['R', 'RESPONDENT' ,'MALE SPEAKER', 'FEMALE SPEAKER', 'INTERVIEWEE', 'THE INTERVIEWEE', 'PARTICIPANT', 'ANSWER']

	#Narratives to start and end at
	first_narrative = 1
	last_narrative = 10

	#Where the folder is
	folder_path = '/Volumes/Research/Adler Research/Sophia OSS Stuff/Narratives_txt/'

		#Loop through all the narratives
	for narrative_number in range(first_narrative,last_narrative):
		print ('reading narrative {narrative_num}'.format(narrative_num = str(narrative_number)))

		narrative_text = readFile(folder_path,narrative_number)

		if narrative_text:
			linesToSpeaker(narrative_text, scenes, interviewer, participant)


if __name__ == '__main__':
	main()
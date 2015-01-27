import os

scenes = ['HIGH POINT', 'LOW POINT', 'TURNING POINT']
interveiwer = ['I']
participant = ['R']


for i in range(1,3):
	print ('reading narrative {narrative_num}'.format(narrative_num = str(i)))
	file_name = 'FLSA_00{narrative_num}.txt'.format(narrative_num = str(i))
	folder_path = '/Volumes/Research/Adler Research/Sophia OSS Stuff/Narratives_txt/'
	full_file_path = folder_path + file_name

	if os.path.isfile(full_file_path):
		narrative_file = open(full_file_path, 'r')

		narrative_text_lines = narrative_file.readlines()
		print (narrative_text_lines)
		for line in narrative_text_lines:
			colon_index = line.find(':')
			if line[:-1].upper() in scenes:
				print ('scene is ' + line)
			elif colon_index == -1:
				print ('speaker is not indicated')
			elif colon_index < 30:
				speaker = line[:colon_index]
				if not(speaker in interveiwer) and not(speaker in participant):
					print (line[:colon_index])
	else:
		print ('Narrative {narrative_num} does not exist'.format(narrative_num = str(i)))
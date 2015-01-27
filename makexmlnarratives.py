import os

#Path variables for narrative location
# -- Just so we don't have to put them on the git
file_name = 'FLSA_001.txt'
folder_path = '/Volumes/Research/Adler Research/Sophia OSS Stuff/'
full_file_path = folder_path + file_name

scenes = ['HIGH POINT', 'LOW POINT', 'TURNING POINT']

print (full_file_path)
print (os.path.isfile(full_file_path))

narrative_file = open(full_file_path, 'r')
#full_narrative_text = narrative_file.read()
narrative_text_lines = narrative_file.readlines()
print (narrative_text_lines)

for line in narrative_text_lines:
	colon_index = line.find(':')
	if line[:-1].upper() in scenes:
		print ('scene is ' + line)
	elif colon_index == -1:
		print ('speaker is not indicated')
	else:
		print (line[:colon_index])
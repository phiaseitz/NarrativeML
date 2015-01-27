import os

#Path variables for narrative location
# -- Just so we don't have to put them on the git
file_name = 'FLSA_001.txt'
folder_path = '/Volumes/Research/Adler Research/Sophia OSS Stuff/'

full_file_path = folder_path + file_name

print (full_file_path)
print (os.path.isfile(full_file_path))

narrative_file = open(full_file_path, 'r')
print (narrative_file.read())
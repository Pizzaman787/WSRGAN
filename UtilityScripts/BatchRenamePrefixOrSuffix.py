#This renames all files in a directory to an incrementing number
from os import listdir
from os.path import isfile, join
from os import rename

path = input("Enter path into folder: ")
prefix = input("Enter prefix: ")
suffix = input("Enter suffix: ")

files = [f for f in listdir(path) if isfile(join(path, f))]
files.sort()

#curIndex = 1

for x in range(len(files)):
	extension = files[x][files[x].index('.'):]
	rename(path + files[x], path + prefix + files[x][0:files[x].index('.')] + suffix + extension)
	#curIndex = curIndex + 1

#This script halves the resolution of files in it and it assumes all files in the path directory are images
from PIL import Image, ImageEnhance # must have installed with pip earlier using pip install pillow
from os import listdir
from os.path import isfile, join

path = input("Enter path into folder: ")
folder = input("Enter folder name to store the results in: ") # folder must already exist

files = [f for f in listdir(path) if isfile(join(path, f))]
#files.sort()

#curIndex = 1

for x in range(len(files)):
	img = Image.open(path + files[x])
	extension = files[x][files[x].index('.'):]
	newsize = (int(img.size[0] / 2), int(img.size[1] / 2))
	img_resized = img.resize(newsize)
	#img_resized.save(folder + "/" + str(curIndex) + extension) # saves the image
	img_resized.save(folder + "/" + files[x])
	#curIndex = curIndex + 1

from PIL import Image, ImageEnhance 
from os import listdir
from os.path import isfile, join


path1 = input("Path to PNG images: ")
folder1 = input("Path to place alphaless PNG images: ")


files1 = [f for f in listdir(path1) if isfile(join(path1, f))]
files1.sort()

for p in range(len(files1)):
	img1 = Image.open(path1 + files1[p])
	img1 = img1.convert('RGB')
	img1.save(folder1 + "/" + str(curIndex) + ".png")

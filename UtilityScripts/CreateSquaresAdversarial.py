
from PIL import Image, ImageEnhance # must have installed with pip earlier using pip install pillow
import numpy as np # must have installed with pip earlier using pip install numpy
from random import randrange
from os import listdir
from os.path import isfile, join

square1Size = 60
square2Size = 60

square1Middle = int((square1Size / 2))
square2Middle = int((square2Size / 2))

square1 = np.zeros((square1Size, square1Size, 4), dtype = np.uint8) # These bug out as Image.fromarray() doesn't know how to read them
square2 = np.zeros((square2Size, square2Size, 4), dtype = np.uint8)
#square1 = np.array(Image.open("5by5Base.png"))
#square2 = np.array(Image.open("7by7Base.png"))

defaultAlpha = 255

path1 = input("Path to upscaled images: ")
path2 = input("Path to original images: ")
folder1 = input("Path to place upscaled squares: ")
folder2 = input("Path to place original squares: ")
repeats = int(input("Do this how many times? "))

if repeats < 0:
	repeats = 0

files1 = [f for f in listdir(path1) if isfile(join(path1, f))]
files1.sort()
files2 = [f for f in listdir(path2) if isfile(join(path2, f))]
files2.sort()

curIndex = 1
tempNum = 0
while tempNum < repeats:
	for p in range(len(files1)): # assumes files1 and files2 are of same size as they were the corresponding images of each other, so p works for both
		img1 = Image.open(path1 + files1[p])
		img2 = Image.open(path2 + files2[p])
		pixels1 = img1.load()
		pixels2 = img2.load()
		extension1 = files1[p][files1[p].index('.'):]
		extension2 = files2[p][files2[p].index('.'):]
		##newsize = (int(img.size[0] / 2), int(img.size[1] / 2))
		##img_resized = img.resize(newsize)
		##img_resized.save(folder + "/" + str(curIndex) + extension) # saves the image
		# gets a random central pixel
		x = randrange(0, img1.size[0], 1)
		y = randrange(0, img1.size[1], 1)
		# have it grab the pixel and add it as the center to the np array
		if (len(pixels1[x, y]) == 4): # for RGBA images
			square1[square1Middle][square1Middle] = pixels1[x, y]
		else: # for taking input from things that are only RGB
			square1[square1Middle][square1Middle] = pixels1[x, y][0], pixels1[x, y][1], pixels1[x, y][2], defaultAlpha
		if (len(pixels2[x, y]) == 4): # for RGBA images
			square2[square2Middle][square2Middle] = pixels2[x, y]
		else: # for taking input from things that are only RGB
			square2[square2Middle][square2Middle] = pixels2[x , y ][0], pixels2[x , y][1], pixels2[x, y][2], defaultAlpha
		# have it go around in a loop filling in the np array if it is in range of the image
		# for square 1
		tempy = -1 * square1Middle #inverses what it takes to get to the center so it is the local cordinate of the top left corner compared to center pixel
		temp1 = 0
		while (temp1 < square1Size): # goes through the whole matrix in row then column
			temp2 = 0
			tempx = -1 * square1Middle
			while (temp2 < square1Size):
				if ((x + tempx) < img1.size[0] and (y + tempy) < img1.size[1]):
					if ((x + tempx) >= 0 and (y + tempy) >= 0):
						if (len(pixels1[x + tempx, y + tempy]) == 4): # for RGBA images
							square1[temp1][temp2] = pixels1[x + tempx, y + tempy]
						else: # for taking input from things that are only RGB
							square1[temp1][temp2] = pixels1[x + tempx, y + tempy][0], pixels1[x + tempx, y + tempy][1], pixels1[x + tempx, y + tempy][2], defaultAlpha
				tempx = tempx + 1
				temp2 = temp2 + 1
			tempy = tempy + 1
			temp1 = temp1 + 1
		# for square 2
		tempy = -1 * square2Middle #inverses what it takes to get to the center so it is the local cordinate of the top left corner compared to center pixel
		temp1 = 0
		while (temp1 < square2Size): # goes through the whole matrix in row then column
			temp2 = 0
			tempx = -1 * square2Middle
			while (temp2 < square2Size):
				if ((x + tempx) < img2.size[0] and (y + tempy) < img2.size[1]):
					if ((x + tempx) >= 0 and (y + tempy) >= 0):
						if (len(pixels2[x + tempx, y + tempy]) == 4): # for RGBA images
							square2[temp1][temp2] = pixels2[x + tempx, y + tempy]
						else: # for taking input from things that are only RGB
							square2[temp1][temp2] = pixels2[x + tempx, y + tempy][0], pixels2[x + tempx, y + tempy][1], pixels2[x + tempx, y + tempy][2], defaultAlpha
				tempx = tempx + 1
				temp2 = temp2 + 1
			tempy = tempy + 1
			temp1 = temp1 + 1
		img_numpy1 = Image.fromarray(square1, mode = "RGBA") # turns the array into an image BROKEN img_numpy1 = Image.fromarray(square1, mode = "RGBA")
		img_numpy2 = Image.fromarray(square2, mode = "RGBA") # turns the array into an image
		img_numpy1.save(folder1 + "/" + str(curIndex) + ".png")
		img_numpy2.save(folder2 + "/" + str(curIndex) + ".png")
		curIndex = curIndex + 1
		square1 = np.zeros((square1Size, square1Size, 4), dtype = np.uint8) # resets the squares
		square2 = np.zeros((square2Size, square2Size, 4), dtype = np.uint8)
	tempNum = tempNum + 1

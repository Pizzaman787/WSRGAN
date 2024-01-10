
from PIL import Image, ImageEnhance # must have installed with pip earlier using pip install pillow
import numpy as np # must have installed with pip earlier using pip install numpy
from random import randrange
from os import listdir
from os.path import isfile, join

inSideSize = 10

square1Middle = int((inSideSize / 2)) #might have to subtract 1 from this to ensure it is actually at the center for indexing

square1 = np.zeros((inSideSize, inSideSize, 4), dtype = np.uint8) # set to zero so uninitiated pixels are transparent
#square1 = np.array(Image.open("5by5Base.png"))

defaultAlpha = 255

path1 = input("Path to image: ")
folder1 = input("Path to place half sized squares: ")

img1 = Image.open(path1)
pixels1 = img1.load()
#pathList = toList(path1)
extension1 = path1[path1.index('.'):]
pieces = []
centers = []
piecesToEdge = int((img1.size[0] / inSideSize) + .9999)

# sets the initial starting x and y position
y = 0
x = 0
xStart = 0
yStart = 0
if (x + square1Middle < img1.size[0]): # tries to optimize the starting square placement if it can
	x = x + square1Middle
	xStart = x
if (y + square1Middle < img1.size[1]):
	y = y + square1Middle
	yStart = y

curIndex = 0
while y < img1.size[1]: #img1.size[1] for height of image
	while x < img1.size[0]: #img1.size[0] for length of image
		# have it grab the pixel and add it as the center to the np array
		if (len(pixels1[x, y]) == 4): # for RGBA images
			square1[square1Middle][square1Middle] = pixels1[x, y]
		else: # for taking input from things that are only RGB
			square1[square1Middle][square1Middle] = pixels1[x, y][0], pixels1[x, y][1], pixels1[x, y][2], defaultAlpha
		# have it go around in a loop filling in the np array if it is in range of the image
		# for square 1
		tempy = -1 * square1Middle #inverses what it takes to get to the center so it is the local cordinate of the top left corner compared to center pixel
		temp1 = 0
		while (temp1 < inSideSize): # goes through the whole matrix in row then column
			temp2 = 0
			tempx = -1 * square1Middle
			while (temp2 < inSideSize):
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
		img_numpy1 = Image.fromarray(square1, mode = "RGBA") # turns the array into an image
		#img_numpy1.save(folder1 + "/" + str(curIndex) + ".png")
		pieces.append(img_numpy1)
		centers.append((x, y))
		curIndex = curIndex + 1
		#print(square1)
		x = x + inSideSize
		square1 = np.zeros((inSideSize, inSideSize, 4), dtype = np.uint8) # resets square1
	y = y + inSideSize
	x = xStart

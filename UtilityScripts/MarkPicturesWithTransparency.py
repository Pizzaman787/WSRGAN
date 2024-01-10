from PIL import Image, ImageEnhance # must have installed with pip earlier using pip install pillow
from os import listdir
from os.path import isfile, join

path = input("Enter path into folder: ")

files = [f for f in listdir(path) if isfile(join(path, f))]
#files.sort()

for x in range(len(files)):
    img = Image.open(path + files[x])
    #extension = files[x][files[x].index('.'):]
    pixels = img.load()
    width = img.size[0]
    height = img.size[1]
    x = 0
    while x < width:
        y = 0
        while y < height:
            if (pixels[x, y][3] < 255): # if there is a transparency
                #need to also have it delete the original before saving a new one
                img.save(path + "TRANSPARENT_" + files[x]) # saves the image with a tag in front of it
            y = y + 1
        x = x + 1

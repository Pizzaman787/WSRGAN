
from os import listdir
from os.path import isfile, join

path = input("The path to the files: ")
files = [f for f in listdir(path) if isfile(join(path, f))]
files.sort()

infront = input("What do you want in front?: ") # change this to whatever you want in front of the text
inback = input("What do you want in back?: ") # change this to whatever you want after the text

textFileName = input("What do you want the text file called?: ")

# makes a text file
f = open(textFileName, "w")

for p in range(len(files)):
    name = "" + infront + files[p] + inback
    f.write(name + "\n")# have it put name and then a new line into the text file

f.close()

import numpy as np
from PIL import Image, ImageEnhance
from random import randrange
import torch
from torch import nn, Tensor
from torchvision.transforms import ToTensor, Lambda, Compose, functional
from os import listdir
from os.path import isfile, join

directory = input("Enter directory of images: ")
outpath = input("Enter directory to store images: ")

files = [f for f in listdir(directory) if isfile(join(directory, f))]
files.sort()

def addAlphaToRGB(img):
    transform = ToTensor()
    inputImage = transform(img)
    temp = inputImage[randrange(0, 3, 1)] # grabs a random non-alpha layer
    inputImage[3] = temp; # sets the alpha layer to be a copy of the randomly selected layer
    output = functional.to_pil_image(inputImage)# converts from a CxHxW tensor to HxWxC image
    return output

for p in range(len(files)):
    inputImage = Image.open(directory + files[p])
    inputImage = inputImage.convert('RGBA')
    inputImage = addAlphaToRGB(inputImage)
    saveName = "alpha" + str(p)
    inputImage.save((outpath + saveName + ".png"), "PNG")

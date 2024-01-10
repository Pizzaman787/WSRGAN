#%matplotlib inline
#import gc
import torch
from torch import reshape
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose, functional
import numpy as np
import torchvision
from torchvision.models import VGG19_Weights
from PIL import Image, ImageEnhance
import time
#import matplotlib.pyplot as plt
#from customDataset import squaresDataset
from os import listdir
from os.path import isfile, join
from random import randrange
import sys
from datetime import datetime
from datetime import timedelta
import math

loadGeneratorOnly = False # set this to true if you are trying to just upscale stuff
batch_size = 32 # How many to run on before applying the optimizer (works by making a tensor of batch_size inputs)
learning_rate = 1e-4
criticPerGenerator = 5
lambdaGP = 10 # gradient penalty value
upscaleFactor = 4
inSideSize = 96
channels = 1
outSideSize = inSideSize * upscaleFactor
inputSize = inSideSize*inSideSize*channels
outputSize = outSideSize*outSideSize*channels
inputPicSize = channels, inSideSize, inSideSize
outputPicSize = channels, outSideSize, outSideSize
savePath = "save_modelNew31OneLayer.pth"
dynamicLearning = False # reduces learning rate every epoch or directory pass
saveToFile = False # whether to save the epoch losses to a txt file
lossSaveFile = "Losses_ModelNew31_OneLayer.txt" # needs to be manually cleared, as is just added to by this program
inputImageType = 'RGB'

# Remember root_dir needs to line up with the expected directory made in the CSV file (don't use a ., so have it be root_dir= dataset and csv says Square1Hold/<filename>)
dataset = 0
train_set, test_set = 0, 0
train_loader = 0
test_loader = 0

def reshuffle():
	global dataset
	global train_set
	global test_set
	global train_loader
	global test_loader
	
	#dataset = squaresDataset(csv_file = 'CSV_File9.csv', root_dir = "dataset/CustomSet7/", transform = ToTensor())

	#train_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset) * .75), len(dataset) - int(len(dataset) * .75)])

	#train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
	#test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# Get cpu or gpu device for training.
device = input("1 for gpu, else cpu: ")
if (device == '1'):
	device = "cuda" if torch.cuda.is_available() else "cpu"
else:
	device = "cpu"
	#torch.set_num_threads(1) # this is used to prevent the parallelism of python from eating all RAM
print("Using {} device".format(device))


class TruncatedVGG19(nn.Module):
	"""
	A truncated VGG19 network, such that its output is the 'feature map obtained by the j-th convolution (after activation)
	before the i-th maxpooling layer within the VGG19 network', as defined in the paper.
	Used to calculate the MSE loss in this VGG feature-space, i.e. the VGG loss.
	"""

	def __init__(self, i, j):
		"""
		:param i: the index i in the definition above
		:param j: the index j in the definition above
		"""
		super(TruncatedVGG19, self).__init__()

		# Load the pre-trained VGG19 available in torchvision
		vgg19 = torchvision.models.vgg19(pretrained=True)

		maxpool_counter = 0
		conv_counter = 0
		truncate_at = 0
		# Iterate through the convolutional section ("features") of the VGG19
		for layer in vgg19.features.children():
			truncate_at += 1

			# Count the number of maxpool layers and the convolutional layers after each maxpool
			if isinstance(layer, nn.Conv2d):
				conv_counter += 1
			if isinstance(layer, nn.MaxPool2d):
				maxpool_counter += 1
				conv_counter = 0

			# Break if we reach the jth convolution after the (i - 1)th maxpool
			if maxpool_counter == i - 1 and conv_counter == j:
				break

		# Check if conditions were satisfied
		assert maxpool_counter == i - 1 and conv_counter == j, "One or both of i=%d and j=%d are not valid choices for the VGG19!" % (
			i, j)

		# Truncate to the jth convolution (+ activation) before the ith maxpool layer
		self.truncated_vgg19 = nn.Sequential(*list(vgg19.features.children())[:truncate_at + 1])

	def forward(self, input):
		"""
		Forward propagation
		:param input: high-resolution or super-resolution images, a tensor of size (N, 3, w * scaling factor, h * scaling factor)
		:return: the specified VGG19 feature map, a tensor of size (N, feature_map_channels, feature_map_w, feature_map_h)
		"""
		output = self.truncated_vgg19(input)  # (N, feature_map_channels, feature_map_w, feature_map_h)

		return output


class ConvolutionalBlock(nn.Module):
	"""
	A convolutional block, comprising convolutional, BN, activation layers.
	"""

	def __init__(self, in_channels, out_channels, kernel_size, stride=1, batch_norm=False, activation=None):
		"""
		:param in_channels: number of input channels
		:param out_channels: number of output channe;s
		:param kernel_size: kernel size
		:param stride: stride
		:param batch_norm: include a BN layer?
		:param activation: Type of activation; None if none
		"""
		super(ConvolutionalBlock, self).__init__()

		if activation is not None:
			activation = activation.lower()
			assert activation in {'prelu', 'leakyrelu', 'tanh'}

		# A container that will hold the layers in this convolutional block
		layers = list()

		# A convolutional layer
		layers.append(
			nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
					  padding=kernel_size // 2))

		# A batch normalization (BN) layer, if wanted
		if batch_norm is True:
			layers.append(nn.BatchNorm2d(num_features=out_channels))

		# An activation layer, if wanted
		if activation == 'prelu':
			layers.append(nn.PReLU())
		elif activation == 'leakyrelu':
			layers.append(nn.LeakyReLU(0.2))
		elif activation == 'tanh':
			layers.append(nn.Tanh())

		# Put together the convolutional block as a sequence of the layers in this container
		self.conv_block = nn.Sequential(*layers)

	def forward(self, input):
		"""
		Forward propagation.
		:param input: input images, a tensor of size (N, in_channels, w, h)
		:return: output images, a tensor of size (N, out_channels, w, h)
		"""
		output = self.conv_block(input)  # (N, out_channels, w, h)

		return output

class SubPixelConvolutionalBlock(nn.Module):
	"""
	A subpixel convolutional block, comprising convolutional, pixel-shuffle, and PReLU activation layers.
	"""

	def __init__(self, kernel_size=3, n_channels=64, scaling_factor=2):
		"""
		:param kernel_size: kernel size of the convolution
		:param n_channels: number of input and output channels
		:param scaling_factor: factor to scale input images by (along both dimensions)
		"""
		super(SubPixelConvolutionalBlock, self).__init__()

		# A convolutional layer that increases the number of channels by scaling factor^2, followed by pixel shuffle and PReLU
		self.conv = nn.Conv2d(in_channels=n_channels, out_channels=n_channels * (scaling_factor ** 2),
							  kernel_size=kernel_size, padding=kernel_size // 2)
		# These additional channels are shuffled to form additional pixels, upscaling each dimension by the scaling factor
		self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scaling_factor)
		self.prelu = nn.PReLU()

	def forward(self, input):
		"""
		Forward propagation.
		:param input: input images, a tensor of size (N, n_channels, w, h)
		:return: scaled output images, a tensor of size (N, n_channels, w * scaling factor, h * scaling factor)
		"""
		output = self.conv(input)  # (N, n_channels * scaling factor^2, w, h)
		output = self.pixel_shuffle(output)  # (N, n_channels, w * scaling factor, h * scaling factor)
		output = self.prelu(output)  # (N, n_channels, w * scaling factor, h * scaling factor)

		return output


class ResidualBlock(nn.Module):
	"""
	A residual block, comprising two convolutional blocks with a residual connection across them.
	"""

	def __init__(self, kernel_size=3, n_channels=64):
		"""
		:param kernel_size: kernel size
		:param n_channels: number of input and output channels (same because the input must be added to the output)
		"""
		super(ResidualBlock, self).__init__()

		# The first convolutional block
		self.conv_block1 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
											  batch_norm=True, activation='PReLu')

		# The second convolutional block
		self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
											  batch_norm=True, activation=None)

	def forward(self, input):
		"""
		Forward propagation.
		:param input: input images, a tensor of size (N, n_channels, w, h)
		:return: output images, a tensor of size (N, n_channels, w, h)
		"""
		residual = input  # (N, n_channels, w, h)
		output = self.conv_block1(input)  # (N, n_channels, w, h)
		output = self.conv_block2(output)  # (N, n_channels, w, h)
		output = output + residual  # (N, n_channels, w, h)

		return output

class SRResNet(nn.Module):
	"""
	The SRResNet, as defined in the paper.
	"""

	def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=upscaleFactor):
		"""
		:param large_kernel_size: kernel size of the first and last convolutions which transform the inputs and outputs
		:param small_kernel_size: kernel size of all convolutions in-between, i.e. those in the residual and subpixel convolutional blocks
		:param n_channels: number of channels in-between, i.e. the input and output channels for the residual and subpixel convolutional blocks
		:param n_blocks: number of residual blocks
		:param scaling_factor: factor to scale input images by (along both dimensions) in the subpixel convolutional block
		"""
		super(SRResNet, self).__init__()

		# Scaling factor must be 2, 4, or 8
		scaling_factor = int(scaling_factor)
		assert scaling_factor in {2, 4, 8}, "The scaling factor must be 2, 4, or 8!"

		# The first convolutional block
		self.conv_block1 = ConvolutionalBlock(in_channels=channels, out_channels=n_channels, kernel_size=large_kernel_size,
											  batch_norm=False, activation='PReLu')

		# A sequence of n_blocks residual blocks, each containing a skip-connection across the block
		self.residual_blocks = nn.Sequential(
			*[ResidualBlock(kernel_size=small_kernel_size, n_channels=n_channels) for i in range(n_blocks)])

		# Another convolutional block
		self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels,
											  kernel_size=small_kernel_size,
											  batch_norm=True, activation=None)

		# Upscaling is done by sub-pixel convolution, with each such block upscaling by a factor of 2
		n_subpixel_convolution_blocks = int(math.log2(scaling_factor))
		self.subpixel_convolutional_blocks = nn.Sequential(
			*[SubPixelConvolutionalBlock(kernel_size=small_kernel_size, n_channels=n_channels, scaling_factor=2) for i
			  in range(n_subpixel_convolution_blocks)])

		# The last convolutional block
		self.conv_block3 = ConvolutionalBlock(in_channels=n_channels, out_channels=channels, kernel_size=large_kernel_size,
											  batch_norm=False, activation=None)

	def forward(self, lr_imgs):
		"""
		Forward prop.
		:param lr_imgs: low-resolution input images, a tensor of size (N, 3, w, h)
		:return: super-resolution output images, a tensor of size (N, 3, w * scaling factor, h * scaling factor)
		"""
		output = self.conv_block1(lr_imgs)  # (N, 3, w, h)
		residual = output  # (N, n_channels, w, h)
		output = self.residual_blocks(output)  # (N, n_channels, w, h)
		output = self.conv_block2(output)  # (N, n_channels, w, h)
		output = output + residual  # (N, n_channels, w, h)
		output = self.subpixel_convolutional_blocks(output)  # (N, n_channels, w * scaling factor, h * scaling factor)
		sr_imgs = self.conv_block3(output)  # (N, 3, w * scaling factor, h * scaling factor)

		return sr_imgs

class Discriminator(nn.Module):
	"""
	The discriminator in the SRGAN, as defined in the paper.
	"""

	def __init__(self, kernel_size=3, n_channels=64, n_blocks=8, fc_size=1024):
		"""
		:param kernel_size: kernel size in all convolutional blocks
		:param n_channels: number of output channels in the first convolutional block, after which it is doubled in every 2nd block thereafter
		:param n_blocks: number of convolutional blocks
		:param fc_size: size of the first fully connected layer
		"""
		super(Discriminator, self).__init__()

		in_channels = channels

		# A series of convolutional blocks
		# The first, third, fifth (and so on) convolutional blocks increase the number of channels but retain image size
		# The second, fourth, sixth (and so on) convolutional blocks retain the same number of channels but halve image size
		# The first convolutional block is unique because it does not employ batch normalization
		conv_blocks = list()
		for i in range(n_blocks):
			out_channels = (n_channels if i is 0 else in_channels * 2) if i % 2 is 0 else in_channels
			conv_blocks.append(
				ConvolutionalBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
								   stride=1 if i % 2 is 0 else 2, batch_norm=i is not 0, activation='LeakyReLu'))
			in_channels = out_channels
		self.conv_blocks = nn.Sequential(*conv_blocks)

		# An adaptive pool layer that resizes it to a standard size
		# For the default input size of 96 and 8 convolutional blocks, this will have no effect
		self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))

		self.fc1 = nn.Linear(out_channels * 6 * 6, fc_size)

		self.leaky_relu = nn.LeakyReLU(0.2)

		self.fc2 = nn.Linear(1024, 1)

		# Don't need a sigmoid layer because the sigmoid operation is performed by PyTorch's nn.BCEWithLogitsLoss()

	def forward(self, imgs):
		"""
		Forward propagation.
		:param imgs: high-resolution or super-resolution images which must be classified as such, a tensor of size (N, 3, w * scaling factor, h * scaling factor)
		:return: a score (logit) for whether it is a high-resolution image, a tensor of size (N)
		"""
		batch_size = imgs.size(0)
		output = self.conv_blocks(imgs)
		output = self.adaptive_pool(output)
		output = self.fc1(output.view(batch_size, -1))
		output = self.leaky_relu(output)
		logit = self.fc2(output)

		return logit
		

class Generator(nn.Module):
	"""
	The generator in the SRGAN, as defined in the paper. Architecture identical to the SRResNet.
	"""

	def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=4):
		"""
		:param large_kernel_size: kernel size of the first and last convolutions which transform the inputs and outputs
		:param small_kernel_size: kernel size of all convolutions in-between, i.e. those in the residual and subpixel convolutional blocks
		:param n_channels: number of channels in-between, i.e. the input and output channels for the residual and subpixel convolutional blocks
		:param n_blocks: number of residual blocks
		:param scaling_factor: factor to scale input images by (along both dimensions) in the subpixel convolutional block
		"""
		super(Generator, self).__init__()

		# The generator is simply an SRResNet, as above
		self.net = SRResNet(large_kernel_size=large_kernel_size, small_kernel_size=small_kernel_size,
							n_channels=n_channels, n_blocks=n_blocks, scaling_factor=scaling_factor)
		# MyVars
		self.timeLearning = timedelta(0) # time in milliseconds spent learning
		self.lastLoss = 0 # loss of the discriminator for this generator on last run
		self.picturesTrainedOn = 0

	def initialize_with_srresnet(self, srresnet_checkpoint):
		"""
		Initialize with weights from a trained SRResNet.
		:param srresnet_checkpoint: checkpoint filepath
		"""
		srresnet = torch.load(srresnet_checkpoint)['model']
		self.net.load_state_dict(srresnet.state_dict())

		print("\nLoaded weights from pre-trained SRResNet.\n")

	def forward(self, lr_imgs):
		"""
		Forward prop.
		:param lr_imgs: low-resolution input images, a tensor of size (N, 3, w, h)
		:return: super-resolution output images, a tensor of size (N, 3, w * scaling factor, h * scaling factor)
		"""
		sr_imgs = self.net(lr_imgs)  # (N, n_channels, w * scaling factor, h * scaling factor)

		return sr_imgs


# used to initialize the networks within wanted range
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv2d') != -1:
		nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		nn.init.normal_(m.weight.data, 1.0, 0.02)
		nn.init.constant_(m.bias.data, 0)
	elif classname.find('InstanceNorm') != -1:
		nn.init.normal_(m.weight.data, 1.0, 0.02)
		nn.init.constant_(m.bias.data, 0)
	
# critic is like the discriminator, real is a tensor of a real image, fake is a tensor of a fake image	
def gradientPenalty(critic, real, fake, device="cpu"):
	BATCH_SIZE, C, H, W = real.shape
	epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
	interpolated_images = real * epsilon + fake * (1 - epsilon)
	
	mixed_scores = critic(interpolated_images)
	gradient = torch.autograd.grad(
		inputs = interpolated_images,
		outputs = mixed_scores, 
		grad_outputs = torch.ones_like(mixed_scores),
		create_graph = True,
		retain_graph = True,
	)[0]
	
	gradient = gradient.view(gradient.shape[0], -1)
	gradient_norm = gradient.norm(2, dim=1)
	gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
	return gradient_penalty

def printLastLoss():
	print(model.last_loss)

def printPicturesTrainedOn():
	print(model.picturesTrainedOn)

def getMomentum(optimizer):
	for g in optimizer.param_groups:
		return(g['momentum'])

def printMomentum(optimizer):
	print(getMomentum(optimizer))

def updateMomentum(optimizer, momentum):
	for g in optimizer.param_groups:
		g['momentum'] = momentum

def getLearningRate(optimizer):
	for g in optimizer.param_groups:
		return(g['lr'])

# ModelNew31_OneLayer.printLearningRate(ModelNew31_OneLayer.optimizerD)
def printLearningRate(optimizer):
		print(getLearningRate(optimizer))
		
# ModelNew31_OneLayer.updateLearningRate(ModelNew31_OneLayer.optimizerD, 0.0001)
def updateLearningRate(optimizer, learningRate):
	for g in optimizer.param_groups:
		g['lr'] = learningRate

# versions of the above functions for easier use
# ModelNew31_OneLayer.updateLearningRateThisModel(0.0001)
def updateLearningRateThisModel(learningRate):
	updateLearningRate(optimizerD, learningRate)
	updateLearningRate(optimizerG, learningRate)
	
def updateMomentumThisModel(momentum):
	updateMomentum(optimizerD, momentum)
	updateMomentum(optimizerG, momentum)

def myCustomLoss(output, target):
	criterion = nn.BCEWithLogitsLoss() # for setting up loss algorithm
	loss = criterion(output, target)
	#loss = myLossAvg(output, target)
	#loss = loss * 1000
	#loss = loss + myLoss3(output, target)
	return loss
	
# for inputing images in manual to see what the loss is
def myCustomLossManual(inputImagePath, targetImagePath):
	transform = ToTensor()
	output = transform(Image.open(inputImagePath))
	target = transform(Image.open(targetImagePath))
	loss = myCustomLoss(output, target)
	loss = loss * 1000
	#loss = loss + myLoss3(output, target)
	return loss.item()

# returns up to 1 for every layer of RBGA wrong (total of 4 if a pixel is completely wrong)
def myLoss(output, target):
	loss = torch.sub(output, target, alpha=1)
	loss = torch.abs(loss)
	loss = torch.sum(loss)
	return loss

def myLossAvg(output, target):
	loss = torch.sub(output, target, alpha=1)
	loss = torch.abs(loss)
	loss = torch.mean(loss)
	return loss
	
# learning rate function
def dynamicLearningRate(lr):
	minNum = 1e-10
	lr = lr * 0.99
	if (lr < minNum):
		lr = minNum
	return lr

def reset():
	global model
	global modelD
	model = Generator().to(device)
	#model.apply(weights_init)
	modelD = Discriminator().to(device)
	#modelD.apply(weights_init)
	global optimizerG
	global optimizerD
	#optimizerG = torch.optim.SGD(model.parameters(),lr=learning_rate, momentum = 0.9, nesterov = False)
	#optimizerD = torch.optim.SGD(modelD.parameters(),lr=learning_rate, momentum = 0.9, nesterov = False)
	optimizerG = torch.optim.Adam(model.parameters(),lr=learning_rate)
	optimizerD = torch.optim.Adam(modelD.parameters(), lr=learning_rate)
	#learning_rate = dynamicLearningRate(model.epoch)

# methods to use on the model ---------------------------------------
def load_weights(model, name):
	pretrained_dict = torch.load(name)
	model_dict = model.state_dict()

	#1. filter out unnecessary keys
	pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

	#2. overwrite entries in the existing state dict
	model_dict.update(pretrained_dict)

	#3. load the new state dict
	model.load_state_dict(model_dict)

# features should be halve sized images of what is the labels
# inspired by "WGAN implementaion from scratch (with gradient penalty)" a YouTube video
def train(net, netD, optimizerG, optimizerD, features, labels, training=True, useNoise=False, device='cpu', trainGen=True, trainDis=True, vgg=True):
	if training:
		netD.train()
		net.train()
	else:
		netD.eval()
		net.eval()
	total_lossG, total_lossD, count = 0, 0, 0
	features, labels = features.to(device), labels.to(device)
	# Converts the 3 layered features/labels into 1 layered versions
	tempRand = randrange(0, 3, 1)
	newFeatures = torch.tensor([[features[0][tempRand].tolist()]]).to(device)
	newLabels = torch.tensor([[labels[0][tempRand].tolist()]]).to(device)
	i = 1
	while i < len(features):
		newFeatures = torch.cat((newFeatures, torch.tensor([[features[i][tempRand].tolist()]]).to(device)), 0).to(device)
		newLabels = torch.cat((newLabels, torch.tensor([[labels[i][tempRand].tolist()]]).to(device)), 0).to(device)
		i = i + 1

	outG = 0
	outD = 0
	lossD = 0
	lossG = 0

	# train Discriminator
	for _ in range(criticPerGenerator):
		if not useNoise:
			outG = net(newFeatures)
		else:
			noise = torch.randn((newFeatures.size(dim=0)), channels, inSideSize, inSideSize, device=device)
			outG = net(noise)
		outD_real = netD(newLabels)
		outD_real = reshape(outD_real, (-1, ))
		outD_fake = netD(outG).reshape(-1)
		outD_fake = reshape(outD_fake, (-1, ))
		gp = gradientPenalty(netD, newLabels, outG, device=device)
		lossD = ( -(torch.mean(outD_real) - torch.mean(outD_fake)) + lambdaGP * gp ) # idea is it is maximizing by negative of minimizing difference
		total_lossD+=float(lossD) # for record keeping
		netD.zero_grad()
		lossD.backward(retain_graph = True)
		if trainDis:
			optimizerD.step()

	total_lossD = total_lossD / criticPerGenerator # for record keeping

	content_loss = 0
	#VGG19 feature part
	if (vgg):
		tempOutG = torch.cat((outG, outG, outG), 1).to(device)
		tempLabels = torch.cat((newLabels, newLabels, newLabels), 1).to(device)

		sr_imgs_in_vgg_space = truncated_vgg19(tempOutG)
		hr_imgs_in_vgg_space = truncated_vgg19(tempLabels).detach()
		content_loss = content_loss_criterion(sr_imgs_in_vgg_space, hr_imgs_in_vgg_space)

	# train Generator
	outD = netD(outG)
	outD = reshape(outD, (-1, ))
	adversarial_loss = myCustomLoss(outD, torch.ones_like(outD))
	#adversarial_loss = (-torch.mean(outD)) # original wasserstien loss idea
	#print(str(content_loss.item()) + "\n")
	#print(str(adversarial_loss.item()) + "\n")
	lossG = (adversarial_loss / 1000) + content_loss
	total_lossG += float(lossG)
	net.zero_grad()
	lossG.backward()
	if trainGen:
		optimizerG.step()

	count+=len(labels)

	# for if to note the loss to a file
	if saveToFile: # appends the lossSaveFile
			with open(lossSaveFile, "a") as f:
				f.write(str(total_lossG / count) + "," + str(total_lossD / count) + "\n")
				f.close()

	# updated images trained on
	net.picturesTrainedOn = net.picturesTrainedOn + len(features)

	return (total_lossG / count, total_lossD / count) # idea is discriminator loss being closer to 0 is good, as 0 means it is 50/50 on whether real or fake

def trainNotGan(net, optimizerG, features, labels, training=True, useNoise=False, device='cpu'):
	if training:
		net.train()
	else:
		net.eval()
	total_lossG, count = 0, 0
	features, labels = features.to(device), labels.to(device)
	# Converts the 3 layered features/labels into 1 layered versions
	tempRand = randrange(0, 3, 1)
	newFeatures = torch.tensor([[features[0][tempRand].tolist()]]).to(device)
	newLabels = torch.tensor([[labels[0][tempRand].tolist()]]).to(device)
	i = 1
	while i < len(features):
		newFeatures = torch.cat((newFeatures, torch.tensor([[features[i][tempRand].tolist()]]).to(device)), 0).to(device)
		newLabels = torch.cat((newLabels, torch.tensor([[labels[i][tempRand].tolist()]]).to(device)), 0).to(device)
		i = i + 1

	outG = 0
	lossG = 0

	# train Generator
	if not useNoise:
		outG = net(newFeatures)
	else:
		noise = torch.randn((newFeatures.size(dim=0)), channels, inSideSize, inSideSize, device=device)
		outG = net(noise)
	adversarial_loss = content_loss_criterion(outG, newLabels)
	#print(str(content_loss.item()) + "\n")
	#print(str(adversarial_loss.item()) + "\n")
	lossG = adversarial_loss
	total_lossG += float(lossG)
	net.zero_grad()
	lossG.backward()
	if training:
		optimizerG.step()

	count+=len(labels)

	# for if to note the loss to a file
	if saveToFile: # appends the lossSaveFile
			with open(lossSaveFile, "a") as f:
				f.write(str(total_lossG / count) + "\n")
				f.close()

	# updated images trained on
	net.picturesTrainedOn = net.picturesTrainedOn + len(features)

	return (total_lossG / count)


def train_epoch(net, netD, dataloader, optimizerG, optimizerD, training = True, useNoise=False, device='cpu', trainGen=True, trainDis=True, vgg=True):
	lossD = 0
	lossG = 0
	count = 0
	for features,labels in dataloader:
		lossGTemp, lossDTemp = train(net, netD, optimizerG, optimizerD, features, labels, training, useNoise, device, trainGen, trainDis, vgg)
		lossG += lossGTemp
		lossD += lossDTemp
		count += 1
	#print("Discriminator:" + str(lossD))
	#print("Generator:" + str(lossG))
	return lossD / count, lossG / count

# ModelNew31_OneLayer.train_DeepLearn2(ModelNew31_OneLayer.model, ModelNew31_OneLayer.modelD, [ModelNew31_OneLayer.Image.open("/home/trace/Pictures/TestFolder/TestImage.png")], ModelNew31_OneLayer.optimizerG, ModelNew31_OneLayer.optimizerD, True)
def train_DeepLearn2(net, netD, imgs, optimizerG=None, optimizerD=None, training = True, random = True, trainGen=True, trainDis=True, vgg=True, gan=True):
	lossD = 0
	lossG = 0
	transform = ToTensor()
	squares = []
	squaresHalved = []
	for i in imgs:
		squareTemp = 0; # placeholder
		if (random): # grabs a random square from the image
			squareTemp = Image.fromarray(grabRandomSquareFromImage(i, inSideSize), mode = "RGBA")
			squareTemp = squareTemp.convert("RGB")# needs to be RGB as vgg doesn't support RGBA
		else: # grabs the center
			squareTemp = Image.fromarray(grabSquareFromImage(i, i.size[0] / 2, i.size[1] / 2, inSideSize), mode = "RGBA")
			squareTemp = squareTemp.convert("RGB")# needs to be RGB as vgg doesn't support RGBA
		newsize = (int(squareTemp.size[0] / upscaleFactor), int(squareTemp.size[1] / upscaleFactor))
		squareHalvedTemp = squareTemp.resize(newsize) # resizes the image

		squareTempAllChannels = torch.tensor(transform(squareTemp).tolist()).to(device)
		squareHalvedTempAllChannels = torch.tensor(transform(squareHalvedTemp).tolist()).to(device)
		# adds squares to the list
		squares.append(squareTempAllChannels.tolist())
		squaresHalved.append(squareHalvedTempAllChannels.tolist())

	features = torch.tensor(squaresHalved).to(device) # needed to add the batch part of the tensor, as it is a tensor of tensors
	labels = torch.tensor(squares).to(device)

	if gan:
		lossGTemp, lossDTemp = train(net, netD, optimizerG, optimizerD, features, labels, training, False, device, trainGen, trainDis, vgg)
		lossG += lossGTemp
		lossD += lossDTemp
	else:
		lossGTemp = trainNotGan(net, optimizerG, features, labels, training, False, device)
		lossG += lossGTemp
		lossD += 0

	return lossD, lossG

# ModelNew31_OneLayer.train_DeepLearnOnDirectory("dataset/train2014/", ModelNew31_OneLayer.model, ModelNew31_OneLayer.modelD, ModelNew31_OneLayer.optimizerG, ModelNew31_OneLayer.optimizerD, True, 1, 0, 0, True, True, True, True, True)
def train_DeepLearnOnDirectory(directory, net, netD, optimizerG=None, optimizerD=None, training = True, batchSize = 1, start=0, limit = 0, random = True, trainGen=True, trainDis=True, vgg=True, gan=True):
	files = [f for f in listdir(directory) if isfile(join(directory, f))]
	files.sort()
	lossD, lossG = 0, 0
	current = 0
	imgs = []
	startTime = datetime.now()
	# determines the number of files/images to be gone over
	if limit > 0 and limit < len(files):
		fileCount = limit
	else:
		fileCount = len(files)

	if start < 0 or start >= fileCount:
		start = 0

	# goes over files/images
	for p in range(fileCount):
		if (p >= start):
			j = (p + 1 - start) / (fileCount - start) # for the percentage printout
			imgs.append(Image.open(("" + directory + files[p])))
			current = current + 1
			if (current >= batchSize):
				tempD, tempG = train_DeepLearn2(net, netD, imgs, optimizerG, optimizerD, training, random, trainGen, trainDis, vgg, gan)
				lossD += tempD
				lossG += tempG
				#resets current and imgs
				current = 0
				imgs.clear()
			# percentage printout
			sys.stdout.write('\r')
			sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j)) # the 20 is number of = in total bar
			sys.stdout.flush()
	print() # ends \r line
	# updates learning rate if dynamicLearning
	if dynamicLearning:
			lr = dynamicLearningRate(getLearningRate(optimizerG))
			updateLearningRate(optimizerG, lr)
			updateLearningRate(optimizerD, lr)
	timeDif = datetime.now() - startTime
	net.timeLearning += timeDif # updates time model has trained
	return lossD / (fileCount / batchSize), lossG / (fileCount / batchSize)

# ModelNew31_OneLayer.train_DeepLearnOnDirectoryRepeat("dataset/train2014/", ModelNew31_OneLayer.model, ModelNew31_OneLayer.modelD, ModelNew31_OneLayer.optimizerG, ModelNew31_OneLayer.optimizerD, True, 1, 16, True, True, True, True, False)
def train_DeepLearnOnDirectoryRepeat(directory, net, netD, optimizerG, optimizerD, training = True, epochs = 1, batchSize = 1, random = True, trainGen=True, trainDis=True, vgg=True, gan=True):
	startTime = datetime.now()
	i = 0
	lossD, lossG = 0, 0
	lossDTotal, lossGTotal = 0, 0
	while i < epochs:
		lossD, lossG = train_DeepLearnOnDirectory(directory, net, netD, optimizerG, optimizerD, training, batchSize, 0, 0, random, trainGen, trainDis, vgg, gan)
		lossDTotal += lossD
		lossGTotal += lossG
		i = i + 1
		print(i)
		print("Discriminator: " + str(lossD))
		print("Generator: " + str(lossG))
	# sets the variables
	net.lastLoss = lossG
	#net.lastLossD = lossD
	timeDif = datetime.now() - startTime
	print("Time running: " + str(timeDif))
	return lossDTotal / i, lossGTotal / i


# SAVING THE MODEL
def saveModel(savePath = savePath):
	torch.save({
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizerG.state_dict(),
			#'epoch': model.epoch,
			'lastLoss': model.lastLoss,
			#'lastLossD': modelD.lastLossD,
			'timeLearning': model.timeLearning,
			'picturesTrainedOn': model.picturesTrainedOn,
			'discrimator_state_dict': modelD.state_dict(),
			'optimizerD_state_dict': optimizerD.state_dict(),
			}, savePath)
	print("Saved PyTorch Model State to " + savePath)

# LOADING A MODEL
def loadModel(savePath = savePath):
	loadModelOnly(savePath)
	loadOptimizersOnly(savePath)

def loadModelG(savePath = savePath):
	global model
	tempDevice = torch.device(device)
	checkpoint = torch.load(savePath, map_location = tempDevice)
	model.load_state_dict(checkpoint['model_state_dict'])
	model.lastLoss = checkpoint['lastLoss']
	model.timeLearning = checkpoint['timeLearning']
	model.picturesTrainedOn = checkpoint['picturesTrainedOn']

def loadModelD(savePath = savePath):
	global modelD
	tempDevice = torch.device(device)
	checkpoint = torch.load(savePath, map_location = tempDevice)
	modelD.load_state_dict(checkpoint['discrimator_state_dict'])
	#modelD.lastLoss = checkpoint['lastLossD']

def loadModelOnly(savePath = savePath):
	loadModelG(savePath)
	loadModelD(savePath)
	print("Loaded model from " + savePath)

def loadOptimizersOnly(savePath = savePath):
	global optimizerG
	global optimizerD
	tempDevice = torch.device(device)
	checkpoint = torch.load(savePath, map_location = tempDevice)
	# optimizers aren't supposed to be on cuda or gpu
	tempDevice = torch.device('cpu')
	checkpoint = torch.load(savePath, map_location = tempDevice)
	optimizerG.load_state_dict(checkpoint['optimizer_state_dict'])
	optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
	print("Loaded optimizers from " + savePath)

# could add tempTensor.to(device) for things to ensure they use the gpu if available
def useOnPILImage(inputImage, cap=False, normalize=False):
	model.eval()
	transform = ToTensor()
	size0 = inputImage.size[0]
	size1 = inputImage.size[1]
	newOutputSize = (channels, size1 * upscaleFactor, size0 * upscaleFactor)
	inputImage = transform(inputImage)
	# this step below is to put the image in into a tensor of tensors to mimic a batch, as Flatten only flattens after the first in the tensor by default
	tensorTemp = torch.tensor([inputImage.tolist()]).to(device)

	# upscales each layer of the input image
	i = 1
	output = model(torch.tensor([[tensorTemp[0][0].tolist()]]).to(device))
	while i < len(tensorTemp[0]):
		output = torch.cat((output, model(torch.tensor([[tensorTemp[0][i].tolist()]]).to(device))), 1).to(device)
		i = i + 1
	output = output[0].to(device)


	#output = output.unflatten(0, newOutputSize)
	# normalizes the image between 0 and 1
	if normalize:
		#tempMin = float(torch.min(output))
		#tempMax = float(torch.max(output))
		#tempMinTen = torch.full(list(output.size()), tempMin, dtype=torch.float, device=device)
		#output = torch.add(output, tempMinTen)
		#output = torch.mul(output, 1 / (tempMin + tempMax))
		output = torch.mul(output, 1 / 2)
		tempAdd = torch.full(list(output.size()), 0.5, dtype=torch.float, device=device)
		output = torch.add(output, tempAdd)
	# caps the image between 0 and 1
	if cap:
		temp1 = torch.ones(newOutputSize).to(device)
		temp2 = torch.zeros(newOutputSize).to(device)
		output = torch.minimum(output, temp1) # acts like ReLU, but caps all above 1
		output = torch.maximum(output, temp2) # acts like ReLU, capping lowest as 0
	output = output.to("cpu") # must be on cpu before converting to a numpy
	output = functional.to_pil_image(output)# converts from a CxHxW tensor to HxWxC image
	return output

def useOnImage(imagePath, cap=False, normalize=False):
	inputImage = Image.open(imagePath)
	inputImage = inputImage.convert(inputImageType)
	img = useOnPILImage(inputImage, cap, normalize)
	return img

# ModelNew31_OneLayer.makeUpscale("TestFolder/563.jpg", cap=True) is an input for testing
def makeUpscale(imagePath, cap=False, normalize=False, outPath="", saveName="tmp.png"):
	img = useOnImage(imagePath, cap, normalize)
	img.save((outPath + saveName))
	del img

def makeUpscaleOfDirectory(directory, cap=False, normalize=False, outPath=""):
	files = [f for f in listdir(directory) if isfile(join(directory, f))]
	files.sort()
	for p in range(len(files)):
		makeUpscsale(("" + directory + files[p]), cap, normalize, outPath, files[p])

def makeImageFromTensor(imgTensor, cap=False, normalize=False, saveName="test.png"):
	testImg = functional.to_pil_image(imgTensor)
	testImg.save(saveName)

# this creates a new image from any input image using the useOnImage function on an image cut into squares before piecing them back together
# it has any overlap created from pieces simply overwritten by the next piece (could make it average between them using a check that likely requires the base be -1 instead of 0)
def useOnFullImage(path, cap):
	square1Middle = int((inSideSize / 2) - 1) #might have to subtract 1 from this to ensure it is actually at the center for indexing

	square1 = np.zeros((inSideSize, inSideSize, 4), dtype = np.uint8) # set to zero so uninitiated pixels are transparent
	#square1 = np.array(Image.open("5by5Base.png"))

	defaultAlpha = 255

	img1 = Image.open(path)
	pixels1 = img1.load()
	extension1 = path[path.index('.'):]
	pieces = []
	pieces2 = []
	centers = []
	piecesToEdge = int((img1.size[0] / inSideSize) + .9999)

	# might need to edit, as images of weird sizes will actually not just be double the size, but centers should still be just double what there is
	# might be able to get by adding up the pieces size
	# could also figure out the size by seeing what padding exists by checking the top left and bottom right corner for what padding could exist (remember center is 5,5 so it is offset a little (5 possible padding above and left, but only 4 right and down). This check can be done by checking the first and last squares made. Then remember to double the size from original image plus padding.
	padx, pady = 0, 0
	newImage = np.full((img1.size[1] * upscaleFactor, img1.size[0] * upscaleFactor, 4), fill_value = 0, dtype = np.uint8) # for storing the new image in numpy array

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
	# splits original image into squares
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
			img_numpy1 = img_numpy1.convert(inputImageType)# Conversion for this model as only uses RGB
			pieces.append(img_numpy1)
			centers.append((x, y))
			curIndex = curIndex + 1
			#print(square1)
			x = x + inSideSize
			#check to see if x is over the size, but less than inSideSize plus the edge, then set it to be at the edge
			if (x >= img1.size[0] and x < img1.size[0] - 1 + inSideSize and x != img1.size[0] + square1Middle):
				x = img1.size[0] - 1
			square1 = np.zeros((inSideSize, inSideSize, 4), dtype = np.uint8) # resets square1
		y = y + inSideSize
		#check to see if y is over the size, but less than inSideSize plus the edge, then set it to be at the edge
		if (y >= img1.size[1] and y < img1.size[1] - 1 + inSideSize and y != img1.size[1] + square1Middle):
			y = img1.size[1] - 1
		x = xStart
	# gets the enlarged pieces
	curIndex = 0
	while curIndex < len(pieces):
		imgTemp = useOnPILImage(pieces[curIndex], cap)
		pieces2.append(imgTemp)
		curIndex = curIndex + 1
	# builds a new image with the enlarged pieces
	padLeft = (centers[0][0] - square1Middle) * -1 * upscaleFactor
	padTop = (centers[0][1] - square1Middle) * -1 * upscaleFactor
	#padRight = (centers[len(centers) - 1][0] - square1Middle) * -1 * 2
	#padBottom = (centers[len(centers) - 1][1] - square1Middle) * -1 * 2
	curIndex = 0
	while curIndex < len(pieces2):
		# use current piece and its corresponding center(multiplied since this is the upscale) to add it to the newImage array
		x = centers[curIndex][0] * upscaleFactor
		y = centers[curIndex][1] * upscaleFactor
		centerOfSquare = square1Middle * upscaleFactor
		#loop through pixels in the pieces
		piecePixels = pieces2[curIndex].load()
		tempy = -1 * square1Middle * upscaleFactor #inverses what it takes to get to the center so it is the local cordinate of the top left corner compared to center pixel
		temp1 = 0
		while (temp1 < (inSideSize * upscaleFactor)): # goes through the whole matrix in row then column
			temp2 = 0
			tempx = -1 * square1Middle * upscaleFactor
			while (temp2 < (inSideSize * upscaleFactor)):
			# TODO: Need to add the padding somewhere
				if ((y + tempy + padTop) < len(newImage) and (x + tempx + padLeft) < len(newImage[0])): # first check for within boundry
					if ((x + tempx + padLeft) >= 0 and (y + tempy + padTop) >= 0): # second check for within boundry
						if (len(piecePixels[centerOfSquare + tempx, centerOfSquare + tempy]) == 4): # for RGBA images (should make this check happen less)
							newImage[y + tempy + padTop][x + tempx + padLeft] = piecePixels[centerOfSquare + tempx, centerOfSquare + tempy]
						else: # for taking input from things that are only RGB
							newImage[y + tempy + padTop][x + tempx + padLeft] = piecePixels[centerOfSquare + tempx, centerOfSquare + tempy][0], piecePixels[centerOfSquare + tempx, centerOfSquare + tempy][1], piecePixels[centerOfSquare + tempx, centerOfSquare + tempy][2], defaultAlpha
				tempx = tempx + 1
				temp2 = temp2 + 1
			tempy = tempy + 1
			temp1 = temp1 + 1
		# increment curIndex
		curIndex = curIndex + 1
	imgOutput = Image.fromarray(newImage, mode = "RGBA") # turns the array into an image
	#Image is for some reason horizontally inverted and rotated 90 degrees, so I just edit it to be correct using the below commands (caused by the numpy array not following same pattern as PIL image)
	#imgOutput = imgOutput.rotate(-90)
	#imgOutput = imgOutput.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
	return imgOutput

# ModelNew31_OneLayer.makeUpscaleOfFullImage("/TestFolder/TestImage.png", cap=True) is an input for testing
def makeUpscaleOfFullImage(imagePath, cap=False, outPath="", saveName="tmp.png"):
	img = useOnFullImage(imagePath, cap)
	img.save((outPath + saveName))

def makeUpscaleOfFullImageDirectory(directory, cap=False, outPath=""):
	files = [f for f in listdir(directory) if isfile(join(directory, f))]
	files.sort()
	for p in range(len(files)):
		makeUpscaleOfFullImage(("" + directory + files[p]), cap, outPath, files[p][0:files[p].index('.')] + ".png")

# returns a tensor of an RGBA image
def grabSquareFromImage(img, x, y, sideLength):
	square1 = np.zeros((sideLength, sideLength, 4), dtype = np.uint8)
	square1Middle = int((sideLength / 2))
	defaultAlpha = 255
	pixels1 = img.load()
	# have it grab the pixel and add it as the center to the np array
	if (type(pixels1[x, y]) == int): # for monochrome images
		square1[square1Middle][square1Middle] = pixels1[x, y], pixels1[x, y], pixels1[x, y], defaultAlpha
	elif (len(pixels1[x, y]) == 4): # for RGBA images
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
			if ((x + tempx) < img.size[0] and (y + tempy) < img.size[1]):
				if ((x + tempx) >= 0 and (y + tempy) >= 0):
					if (type(pixels1[x, y]) == int): # for monochrome images
						square1[temp1][temp2] = pixels1[x + tempx, y + tempy], pixels1[x + tempx, y + tempy], pixels1[x + tempx, y + tempy], defaultAlpha
					elif (len(pixels1[x + tempx, y + tempy]) == 4): # for RGBA images
						square1[temp1][temp2] = pixels1[x + tempx, y + tempy]
					else: # for taking input from things that are only RGB
						square1[temp1][temp2] = pixels1[x + tempx, y + tempy][0], pixels1[x + tempx, y + tempy][1], pixels1[x + tempx, y + tempy][2], defaultAlpha
			tempx = tempx + 1
			temp2 = temp2 + 1
		tempy = tempy + 1
		temp1 = temp1 + 1
	#img_numpy1 = Image.fromarray(square1, mode = "RGBA") # turns the array into an image
	return square1

def grabRandomSquareFromImage(img, sideLength):
	#img = Image.open(imgPath)
	x = randrange(0, img.size[0], 1)
	y = randrange(0, img.size[1], 1)
	return grabSquareFromImage(img, x, y, sideLength)

# stuff for setting up model to run ---------------------------------------------------
reshuffle()
beta1 = 0.5 # for setting up the optimizers
real_label = 1.
fake_label = 0.
#below stuff is made to be overwritten by reset()
model = 0
modelD = 0
optimizerG = 0
optimizerD = 0
loss_fn = myCustomLoss
learning_rate = dynamicLearningRate(learning_rate)
truncated_vgg19 = 0
content_loss_criterion = 0
if (not loadGeneratorOnly):
	reset() # loads the optimizer and model
	print(model)
	#print(modelD)
	# VGG setup
	truncated_vgg19 = TruncatedVGG19(i=5, j=4)
	truncated_vgg19.eval()
	truncated_vgg19 = truncated_vgg19.to(device)
	content_loss_criterion = nn.MSELoss()
	content_loss_criterion = content_loss_criterion.to(device)
else:
	model = Generator().to(device)
	loadModelG()

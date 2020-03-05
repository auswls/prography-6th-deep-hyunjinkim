import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.pyplot import imread
import numpy as np
from skimage import transform
import os
import tarfile
import urllib

class VGG16(nn.Module):
	def __init__(self):
		super(VGG16, self).__init__()
		# conv layers: (in_channel size, out_channels size, kernel_size, stride, padding)
		self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
		self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

		self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
		self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

		self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
		self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

		self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
		self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

		self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

		# skip connection
		self.skip = nn.Conv2d(64, 512, kernel_size = 3, padding = 1, stride = 18)

		# max pooling (kernel_size, stride)
		self.pool = nn.MaxPool2d(2, 2)

		# fully connected layers:
		self.fc6 = nn.Linear(7*7*512*2, 4096)
		self.fc7 = nn.Linear(4096, 4096)
		self.fc8 = nn.Linear(4096, 10) # number of mnist class = 10

	def forward(self, x, training=True):
		x = F.relu(self.conv1_1(x))
		x = F.relu(self.conv1_2(x))
		x = self.pool(x)
		
		conv2_1_input = x
		skip_connection = self.skip(x)
		#print("conv2_1 input size is ", conv2_1_input.size())
		#print("skip connection size is ", skip_connection.size())

		x = F.relu(self.conv2_1(x))
		x = F.relu(self.conv2_2(x))
		x = self.pool(x)

		x = F.relu(self.conv3_1(x))
		x = F.relu(self.conv3_2(x))
		x = F.relu(self.conv3_3(x))
		x = self.pool(x)

		x = F.relu(self.conv4_1(x))
		x = F.relu(self.conv4_2(x))
		x = F.relu(self.conv4_3(x))
		x = self.pool(x)

		x = F.relu(self.conv5_1(x))
		x = F.relu(self.conv5_2(x))
		x = F.relu(self.conv5_3(x))
		x = self.pool(x)
		
		#print("x.size() is ", x.size())

		# add skip connection to input
		x = torch.cat((x, skip_connection), dim = 1)

		x = x.view(-1, 7*7*512*2)
		#x = x.view(x.size(0), 7*7*512*2)

		x = F.relu(self.fc6(x))
		x = F.dropout(x, 0.5, training=training)
		x = F.relu(self.fc7(x))
		x = F.dropout(x, 0.5, training=training)
		x = self.fc8(x)
		
		return x


"""
##### for test #####
model = VGG16()

A=torch.ones([1,3,224,224])

result = model(A)
#print(result)
#print(result.size())
"""

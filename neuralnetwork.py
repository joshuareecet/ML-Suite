import torch
from torch import nn
import torch.nn.functional as F
from utils.setup import device

class NeuralNetwork(nn.Module):
	def __init__(self, num_classes = 10):
		super().__init__()
		self.flatten = nn.Flatten()

		self.linear_relu_stack = nn.Sequential(
			nn.Linear(28*28, 512),
			nn.ReLU(),
			nn.Linear(512,512),
			nn.ReLU(),
			nn.Linear(512,num_classes)
		)

	def forward(self, x):
		x = self.flatten(x)
		logits = self.linear_relu_stack(x)
		return logits
	
class CNN(nn.Module):
	def __init__(self, in_channels = 1, num_classes = 10, imgsz = (28,28)):
		super().__init__()
		self.flatten = nn.Flatten(1,3)

		conv1_out = 32
		self.conv1 = nn.Conv2d(in_channels,conv1_out,kernel_size=3,stride=1,padding=1)
		self.bn1 = nn.BatchNorm2d(conv1_out)
		
		self.conv2 = nn.Conv2d(conv1_out,conv1_out*2,kernel_size=3,stride=1,padding=1)
		self.bn2 = nn.BatchNorm2d(conv1_out*2)
		
		self.conv3 = nn.Conv2d(conv1_out*2,conv1_out*2,kernel_size=3,stride=1,padding=1)
		self.bn3 = nn.BatchNorm2d(conv1_out*2)
		
		self.halving_pool = nn.MaxPool2d(kernel_size=2,stride=2)
		self.global_pool = nn.MaxPool2d(kernel_size=2,stride=2)

		gp_imgsz = (int(imgsz[0]/4), int(imgsz[1]/4))
		self.conv_dropout = nn.Dropout(0.1)
		self.fc_dropout = nn.Dropout(0.2)

		fc_in_sz = int(conv1_out*2*(gp_imgsz[0] * gp_imgsz[1]))
		self.fully_connected = nn.Linear(fc_in_sz,256)
		self.fully_connected2 = nn.Linear(256,num_classes)

	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = self.halving_pool(x)
		x = self.conv_dropout(x)
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		x = self.global_pool(x)
		x = self.flatten(x)
		x = self.fc_dropout(x)
		x = self.fully_connected(x)
		x = F.relu(x)
		x = self.fully_connected2(x)
		return x


if __name__ == "__main__":
	model = NeuralNetwork().to(device)
	print(model)

	model = CNN().to(device)
	print(model)
	
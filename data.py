import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.io import decode_image
import pandas as pd
import os
from utils.setup import data_dir
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
default_test_transform = [
		v2.ToImage(),
		v2.ToDtype(torch.float32,scale=True),
		v2.Normalize(mean=[0.2860], std=[0.3530])
	]
default_train_transform = [
		*default_test_transform,
		v2.RandomRotation(6),
		v2.RandomHorizontalFlip(0.15),
		v2.RandomCrop(28,padding=4)
	]

# Custom Dataset Classes ---------------------------------------------------------------------------------------------------------

class CustomImageDataset(Dataset):
	"""Image dataset backed by a CSV annotations file and an image directory."""

	def __init__(self, annotations_file, img_dir, transform = None, target_transform = None):
		self.img_dir = img_dir
		self.img_labels = pd.read_csv(annotations_file)
		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return len(self.img_labels)

	def __getitem__(self, idx):
		img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx,0])
		image = decode_image(img_path)
		label = self.img_labels.iloc[idx,1]
		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			label = self.target_transform(label)
		return image, label
	
class TransformedSubset(Dataset):
	"""Wraps a Subset and applies separate transforms to images and labels."""

	def __init__(
			self, 
			subset: Subset,
			transform: v2.Transform = None,
			target_transform: v2.Transform = None
	):
		self.data = subset
		self.transform = transform
		self.target_transform = target_transform

	def __len__(self) -> int:
		return len(self.data)
	
	def __getitem__(self, idx):
		img, label = self.data[idx]
		if self.transform:
			img = self.transform(img)
		if self.target_transform:
			label = self.target_transform(label)
		return img,label
	
	def add_transform(self, trans: v2.Transform):
		self.transform = trans
	def add_target_transform(self, trans: v2.Transform):
		self.target_transform = trans

# Custom Dataset Splitters ---------------------------------------------------------------------------------------------------------
def get_labels(dataset: Dataset) -> torch.Tensor:
	"""Extract labels from a dataset, falling back to iterating if no targets attribute exists."""
	if hasattr(dataset, "targets"):
		return dataset.targets
	elif hasattr(dataset, "labels"):
		return dataset.labels
	else:
		return torch.tensor([dataset[i][1] for i in range(len(dataset))])

def train_val_split(
		dataset: Dataset,
		stratified=False, 
		val_size = 0.2,
		train_transform = v2.Compose(default_train_transform),
		val_transform = v2.Compose(default_test_transform),
) -> tuple[TransformedSubset, TransformedSubset]:
	"""Split a dataset into train/val TransformedSubsets, optionally stratified by class."""
	labels = get_labels(dataset) if stratified else None
	
	train_idx, val_idx = train_test_split(
		range(len(dataset)),
		test_size=val_size,
		stratify=labels,
		random_state=RANDOM_SEED
	)

	train_subset = TransformedSubset(Subset(dataset, train_idx), transform=train_transform)
	val_subset = TransformedSubset(Subset(dataset, val_idx), transform=val_transform)
	
	return train_subset, val_subset

# Loading data ---------------------------------------------------------------------------------------------------------

def get_FMNIST_data() -> tuple[TransformedSubset,TransformedSubset,Dataset]:
	"""Load FashionMNIST and return stratified (train, val, test) splits."""
	FMNIST_data = datasets.FashionMNIST(
		root = data_dir,
		train=True,
		download=True,
	)

	FMNIST_training_data, FMNIST_val_data = train_val_split(FMNIST_data, stratified=True)

	FMNIST_test_data = datasets.FashionMNIST(
		root = data_dir,
		train=False,
		download=True,
		transform=v2.Compose(default_test_transform)
	)

	return FMNIST_training_data, FMNIST_val_data, FMNIST_test_data
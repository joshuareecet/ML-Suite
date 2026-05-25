import json
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.io import decode_image
import pandas as pd
import os
from utils.setup import data_dir
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
CONFIG_PATH = data_dir / "dataset_config.json"


def build_test_transform(mean: list[float], std: list[float]) -> v2.Compose:
	return v2.Compose([
		v2.ToImage(),
		v2.ToDtype(torch.float32, scale=True),
		v2.Normalize(mean=mean, std=std)
	])

def build_train_transform(mean: list[float], std: list[float], imgsz: tuple[int, int]) -> v2.Compose:
	return v2.Compose([
		v2.ToImage(),
		v2.ToDtype(torch.float32, scale=True),
		v2.Normalize(mean=mean, std=std),
		v2.RandomRotation(6),
		v2.RandomHorizontalFlip(0.15),
		v2.RandomCrop(imgsz, padding=4)
	])


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
		train_transform: v2.Compose = None,
		val_transform: v2.Compose = None,
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

def get_normalisation(dataset: type[datasets.VisionDataset]) -> tuple[list[float], list[float], tuple[int, int]]:
	"""Return (mean, std, imgsz) for the dataset, computing and caching in dataset_config.json if needed."""
	dataset_name = dataset.__name__

	config = {}
	if CONFIG_PATH.exists():
		with open(CONFIG_PATH) as f:
			config = json.load(f)

	entry = config.get(dataset_name, {})
	if "mean" in entry and "std" in entry and "imgsz" in entry:
		return entry["mean"], entry["std"], tuple(entry["imgsz"])

	print(f"Computing normalisation stats for {dataset_name}...")
	raw = dataset(root=data_dir, train=True, download=True, transform=v2.Compose([
		v2.ToImage(),
		v2.ToDtype(torch.float32, scale=True)
	]))
	loader = DataLoader(raw, batch_size=512, num_workers=2)

	total = None
	total_sq = None
	count = 0
	imgsz = None

	for imgs, _ in loader:
		b, c, h, w = imgs.shape
		if total is None:
			total = torch.zeros(c)
			total_sq = torch.zeros(c)
			imgsz = (h, w)
		total += imgs.sum(dim=[0, 2, 3])
		total_sq += (imgs ** 2).sum(dim=[0, 2, 3])
		count += b * h * w

	mean = (total / count).tolist()
	std = ((total_sq / count - torch.tensor(mean) ** 2).clamp(min=0) ** 0.5).tolist()

	config[dataset_name] = {"mean": mean, "std": std, "imgsz": list(imgsz)}
	with open(CONFIG_PATH, "w") as f:
		json.dump(config, f, indent=2)

	return mean, std, imgsz

def get_dataset(dataset: type[datasets.VisionDataset] = datasets.FashionMNIST) -> tuple[TransformedSubset, TransformedSubset, Dataset]:
	mean, std, imgsz = get_normalisation(dataset)

	data = dataset(root=data_dir, train=True, download=True)
	train_data, val_data = train_val_split(
		data,
		stratified=True,
		train_transform=build_train_transform(mean, std, imgsz),
		val_transform=build_test_transform(mean, std),
	)

	test_data = dataset(
		root=data_dir,
		train=False,
		download=True,
		transform=build_test_transform(mean, std)
	)

	return train_data, val_data, test_data

def get_dataset_info(train_data: TransformedSubset) -> tuple[int, int, tuple[int, int]]:
	"""Return (in_channels, num_classes, imgsz) by inspecting a sample and the underlying dataset."""
	sample_x, _ = train_data[0]
	in_channels, h, w = sample_x.shape
	raw_dataset = train_data.data.dataset
	if hasattr(raw_dataset, "classes"):
		num_classes = len(raw_dataset.classes)
	else:
		num_classes = len(set(get_labels(raw_dataset).tolist()))
	return in_channels, num_classes, (h, w)
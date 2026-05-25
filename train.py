import torch
from torch import nn
from torch.utils.data import DataLoader

from data import get_dataset
from torchvision import datasets

from neuralnetwork import SimpleMLP, SimpleCNN, StridedCNN, MiniResNet, Res50
from utils.setup import device, models_dir
import time
# Training hyperparameters -----------------------------------------------------------------------------------------------------
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 50

MODEL = Res50
DATASET = datasets.FashionMNIST
DATASET_NAME = f"FMNIST"

DEBUG = False
if DEBUG:
	torch.autograd.set_detect_anomaly(True)

# Train and test loops ---------------------------------------------------------------------------------------------------------
def train_loop(
		dataloader: DataLoader,
		model: torch.nn.Module,
		loss_fn: torch.nn.Module,
		optimizer: torch.optim.Optimizer,
):
	"""Run one epoch of training. Returns mean training loss."""
	train_loss = 0
	size = len(dataloader.dataset)
	num_batches = len(dataloader)
	model.train()
	
	for batch, (X,Y) in enumerate(dataloader):
		# make sure we use channels last format
		X = X.to(device, memory_format=torch.channels_last)
		Y = Y.to(device)
		
		pred = model(X)
		loss = loss_fn(pred, Y)
		train_loss += loss.item()

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if batch % 100 == 0:
			curr_loss, cur_progress = loss.item(), batch * BATCH_SIZE + len(X)
			print(f"Current loss: {curr_loss:>7f}	|	[{cur_progress:>5d} / {size:>5d}]")
	
	train_loss /= num_batches
	return train_loss

def val_loop(dataloader: DataLoader, model: torch.nn.Module, loss_fn: torch.nn.Module):
	"""Evaluate the model on a dataloader. Returns (mean loss, accuracy)."""
	model.eval()
	size = len(dataloader.dataset)
	num_batches = len(dataloader)
	test_loss, correct = 0, 0

	with torch.no_grad():
		for X, Y in dataloader:
			# make sure we use channels last format
			X = X.to(device, memory_format=torch.channels_last)
			Y = Y.to(device)

			y = model(X)
			test_loss += loss_fn(y, Y).item()
			correct += (y.argmax(1) == Y).type(torch.float).sum().item()
	
	test_loss /= num_batches
	correct /= size
	return test_loss, correct

# Model initialiser functions ---------------------------------------------------------------------------------------------------------
def load_model(model: nn.Module):
	"""Load saved weights into model from model_name path. No-op if file doesn't exist."""
	model_load_path = models_dir / f"{DATASET_NAME}{model.model_name()}_best.pt"
	try:
		if model_load_path.exists():
			model.load_state_dict(torch.load(model_load_path,weights_only=True))
		else:
			print("No prev model loaded..")
	except Exception as e:
		print(e)
		print(f"Couldn't load model: {DATASET_NAME}{model.model_name()}_best.pt \nAt path: {model_load_path}")
		raise e

def model_init(
		model_type: nn.Module,
		loss_fn_ = nn.CrossEntropyLoss, 
		optimizer_ = torch.optim.AdamW, 
		scheduler_ = torch.optim.lr_scheduler.LinearLR
):
	"""Build and return (model, loss_fn, optimizer, scheduler) for CNN."""
	model = model_type().to(device).to(memory_format=torch.channels_last)
	load_model(model)

	loss_fn = loss_fn_()
	optimizer = optimizer_(params=model.parameters(),lr=LEARNING_RATE)
	scheduler = scheduler_(optimizer, start_factor=1.0, end_factor=0.5, total_iters=30)
	
	return model, loss_fn, optimizer, scheduler

# Train loop -----------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
	
	# Loading Data 
	train_data, val_data, test_data = get_dataset(DATASET)

	train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=2, persistent_workers=True, shuffle=True)
	val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=2, persistent_workers=True)
	test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=2, persistent_workers=True)
	
	# Initialising Model
	model, loss_fn, optimizer, scheduler = model_init(MODEL)
	save_path = models_dir / f"{DATASET_NAME}{model.model_name()}_best.pt"
	
	print(f"Using device: {device}")
	best_loss = None
	# Run Training 
	for t in range(EPOCHS):
		print(f"Epoch {t+1}\n-------------------------------")
		train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
		valid_loss, correct = val_loop(val_dataloader, model, loss_fn)
		scheduler.step()
		print(f"Validation Error: \n Accuracy: {(100*correct):>0.1f}% \nTrain loss: {train_loss:>8f} \nValid loss: {valid_loss:>8f} \n")
		if (best_loss is None) or (valid_loss < best_loss):
			torch.save(model.state_dict(), save_path)
			best_loss = valid_loss

	print("Done!")
	
	# Evaluate on test set 
	test_loss, correct = val_loop(test_dataloader, model, loss_fn)
	print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
	torch.save(model.state_dict(), models_dir / f"{DATASET_NAME}_last.pt")
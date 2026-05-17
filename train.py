import torch
from torch import nn
from torch.utils.data import DataLoader

from data import get_FMNIST_data
from neuralnetwork import NeuralNetwork, CNN
from utils.setup import device, models_dir

# Training hyperparameters -----------------------------------------------------------------------------------------------------
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
EPOCHS = 50
MODEL_NAME = "FMNIST_MODEL_CONV_V2"
model_name = models_dir / MODEL_NAME

# Train and test loops ---------------------------------------------------------------------------------------------------------
def train_loop(
		dataloader: DataLoader, 
		model: torch.nn.Module, 
		loss_fn: torch.nn.Module, 
		optimizer: torch.optim.Optimizer,
		scheduler: torch.optim.lr_scheduler.LRScheduler = None
):
	size = len(dataloader.dataset)
	model.train()
	for batch, (X,Y) in enumerate(dataloader):
		# make sure we use channels last format
		X = X.to(device, memory_format=torch.channels_last)
		Y = Y.to(device)
		
		pred = model(X)
		loss = loss_fn(pred, Y)
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if batch % 100 == 0:
			curr_loss, cur_progress = loss.item(), batch * BATCH_SIZE + len(X)
			print(f"Current loss: {curr_loss:>7f}	|	[{cur_progress:>5d} / {size:>5d}]")
	if scheduler: scheduler.step()

def val_loop(dataloader: DataLoader, model: torch.nn.Module, loss_fn: torch.nn.Module):
	model.eval()
	size = len(dataloader.dataset)
	num_batches = len(dataloader)
	test_loss, correct = 0, 0

	with torch.no_grad():
		for X, y in dataloader:
			# make sure we use channels last format
			X = X.to(device, memory_format=torch.channels_last)
			y = y.to(device)

			Y = model(X)
			test_loss += loss_fn(Y, y).item()
			correct += (Y.argmax(1) == y).type(torch.float).sum().item()
	
	test_loss /= num_batches
	correct /= size
	return test_loss, correct

# Model initialiser functions ---------------------------------------------------------------------------------------------------------
def load_model(model: nn.Module):
	try:
		if model_name.exists():
			model.load_state_dict(torch.load(model_name,weights_only=True))
		else:
			print("No prev model loaded..")
	except Exception as e:
		print(e)
		print(f"Couldn't load model: {MODEL_NAME}	\nAt path: {model_name}")
		raise e
	
def neural_net_init():
	model = NeuralNetwork().to(device)
	model = model.to(memory_format=torch.channels_last)
	load_model(model)
	
	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(),lr=LEARNING_RATE)
	scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=30)
	
	return model, loss_fn, optimizer, scheduler

def CNN_init():
	model = CNN().to(device)
	model = model.to(memory_format=torch.channels_last)
	load_model(model)

	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.AdamW(params=model.parameters(),lr=LEARNING_RATE)
	scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=30)
	
	return model, loss_fn, optimizer, scheduler


# Train loop -----------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
	# Loading Data 
	FMNIST_training_data, FMNIST_val_data, FMNIST_test_data = get_FMNIST_data()

	train_dataloader = DataLoader(FMNIST_training_data, batch_size=BATCH_SIZE, num_workers=2, persistent_workers=True, shuffle=True)
	val_dataloader = DataLoader(FMNIST_val_data, batch_size=BATCH_SIZE, num_workers=2, persistent_workers=True)
	test_dataloader = DataLoader(FMNIST_test_data, batch_size=BATCH_SIZE, num_workers=2, persistent_workers=True)
	
	# Initialising Model
	model, loss_fn, optimizer, scheduler = CNN_init()
	print(f"Using device: {device}")

	# Run Training 
	for t in range(EPOCHS):
		print(f"Epoch {t+1}\n-------------------------------")
		train_loop(train_dataloader, model, loss_fn, optimizer, scheduler)
		test_loss, correct = val_loop(val_dataloader, model, loss_fn)
		print(f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
	print("Done!")
	
	# Evaluate on test set 
	test_loss, correct = val_loop(test_dataloader, model, loss_fn)
	print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
	torch.save(model.state_dict(), model_name)
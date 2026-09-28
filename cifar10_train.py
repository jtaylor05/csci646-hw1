from pathlib import Path
import argparse

import torch, time, wandb
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision.transforms import transforms

from torchmetrics.classification import MulticlassF1Score

import numpy as np

from utils import str2list, str2activation
from cifar10_model import CNNModel
from cifar10_resnet import ResNet_18

parser = argparse.ArgumentParser(description = "CIFAR-10 CNN Training")
parser.add_argument("--mode", type=str, default='train', help="train or test")
parser.add_argument("--epochs", type=int, default=40, help="Num of epochs, aim for range between [10,50]")

parser.add_argument("--out_channels", dest="outc", type=str2list, default=[32,32], help="Out channels, len(in) = 1 || layer_num. (ex. 12,16)")
parser.add_argument("--kernel_size", dest="ksize", type=str2list, default=3, help="Size of all kernels, len(in) = 1 || layer_num. (ex. 3,5)")
parser.add_argument("--stride", dest="stride", type=str2list, default=1, help="Stride of all operations, len(in) = 1 || layer_num. (ex. 1,2)")
parser.add_argument("--padding", dest="pad", type=str2list, default=0, help="Padding of all operations, len(in) = 1 || layer_num. (ex. 0,3)")
parser.add_argument("--output_size", dest="outs", type=str2list, default=[40], help="Output size of fully connected layers. There will be len(output_size) + 1 layers.")

parser.add_argument("--classes", type=int, default=10, help="Number of output classes.")
parser.add_argument("--pooling", type=int, default=None, help="Kernel size pooling. Leave empty for none.")
parser.add_argument("--avg_pool", action="store_true", help="Whether to use AvgPool2d. Otherwise uses MaxPool2d")
parser.add_argument("--dropout", type=float, default=None, help="Dropout to use each layer. Leave empty for none.")
parser.add_argument("--norm", action="store_true", help="Whether or not to use batch normalization each layer. Leave empty for none.")
parser.add_argument("--activation", type=str2activation, default=nn.ReLU, help="Type of Activation function. Valid: 'relu', 'elu', 'lrelu', 'tanh', 'sigmoid'.")

parser.add_argument("--learning_rate", type=float, default=0.001, help = "Learning rate for model parameters.")
parser.add_argument("--decay", type=float, default=0.5, help = "Decay used in LR scheduling.")
parser.add_argument("--batch_size", type=int, default=100, help="batch size")
parser.add_argument("--rotation", dest="rotation", type=int, default=10, help="image rotation during preprocessing.")
parser.add_argument("--load_checkpoint", action="store_true", help="Whether or not to load checkpoint from ckp_path")

parser.add_argument("--ckp_path", type=str, default="data/checkpoint", help="Path to checkpoint data.")
parser.add_argument("--wandb_name", type=str, default="CIFAR10", help="What name to log this run as on WandB.")
parser.add_argument("--resnet", type=bool, default=None, help="Whether to use the ResNet-18 architecture. True for normal, false for skipping.")

args = parser.parse_args()

def _get_dataloader(data_path: str, batch_size: int, rotation=10):	
	train_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(degrees=(rotation, rotation)),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))
    ])
	train_dataset = datasets.CIFAR10(root=data_path, download=True, train=True, transform=train_trans)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
	
	## for testing
	test_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
	test_dataset = datasets.CIFAR10(root=data_path, download=True, train=False, transform=test_trans)
	test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
	
	return train_loader, test_loader


def _save_checkpoint(path, model: nn.Module, epoch, optimizer: optim.Optimizer, global_step):
    checkpoint = {
		'epoch': epoch,
		'global_step': global_step,
		'model': model.state_dict(),
		'optimizer': optimizer.state_dict()
	}
    torch.save(checkpoint, path)

def _load_checkpoint(path, model: nn.Module, optimizer: optim.Optimizer):
    if not Path(path).exists():
        return 0, 0
    
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch'], checkpoint['global_step']

def _compute_accuracy(y_pred, y_batch):
	## please write the code below ##
	accy = (y_pred == y_batch).sum().item() / y_batch.size(0)
	return accy

def _init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)

def main():

	use_cuda = torch.cuda.is_available() ## if have gpu or cpu 
	device = torch.device("cuda" if use_cuda else "cpu")
	print("device: ", device)
	if use_cuda:
		torch.cuda.manual_seed(72)

	n_epochs = args.epochs
	decay = args.decay
	learning_rate = args.learning_rate

	DATA_PATH = "./data/"
	train_loader, test_loader=_get_dataloader(DATA_PATH, args.batch_size, args.rotation)
 
	use_pooling = args.pooling is not None
	p_kernel = args.pooling if use_pooling else 0
	pool_type = nn.AvgPool2d if args.avg_pool else nn.MaxPool2d
 
	use_dropout = args.dropout is not None
	dropout = args.dropout if use_dropout else 0
	
	# Dynamic CNNModel defined in cifar10_model.py
	model = CNNModel(
		args.outc,   # out channels
		args.ksize,  # kernel size
		args.stride, # stride
		args.pad,    # padding
		args.outs,   # output size
		classes=args.classes,
		use_dropout=use_dropout, dropout=dropout,
		use_pooling=use_pooling, pool_class=pool_type, pool_kernel=p_kernel,
		use_batch_norm=args.norm, activation=args.activation
    ) 
 
	if args.resnet is not None:
		model = ResNet_18(args.resnet)
 
	model.apply(_init_weights)
	
	# TODO: add DP

	model.to(device)
	
	# TODO: Dynamic lossfunction/learning_rate
	optimizer = optim.Adam(model.parameters(),lr=learning_rate)  ## optimizer
	loss_fun = nn.CrossEntropyLoss()
	scheduler = optim.lr_scheduler.ExponentialLR(optimizer, decay)
	
	# TODO: Improve look of checkpointing. Check that it works.
	epoch = 0
	global_step = 0
	if args.load_checkpoint == True:
		epoch, global_step = _load_checkpoint(args.ckp_path, model, optimizer)

	metric = MulticlassF1Score(args.classes, average='macro').to(device)
	
	##  Train Model
	if args.mode == 'train':
		model = model.train()
		for epoch in range(epoch, n_epochs):

			for batch_id, (x_batch,y_labels) in enumerate(train_loader):
				x_batch,y_labels = Variable(x_batch).to(device), Variable(y_labels).to(device)

				# feed input data x into model
				output_y: torch.Tensor = model(x_batch)
				
				# Use CrossEntropyLoss function.
				loss = loss_fun(output_y, y_labels)
				

				# Back propagation
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				# Get Accuracy
				_, y_pred = torch.max(output_y, 1)
				accy = _compute_accuracy(y_pred, y_labels)
    
				# Get F1
				f1 = metric(y_pred, y_labels)
				
				# Log on WandB
				wandb.log({
					'epoch': epoch,
					'loss' : loss.item(),
					'acc'  : accy,
					'f1'   : f1
				})
				global_step+=1

			# Save checkpoints to ckp_path
			_save_checkpoint(args.ckp_path, model, epoch, optimizer, global_step)

		scheduler.step()
				

	acc_list = []
	f1_list = []
	# Test Model
	model.eval()
	with torch.no_grad():
		for batch_id, (x_batch,y_labels) in enumerate(test_loader):
			x_batch, y_labels = Variable(x_batch).to(device), Variable(y_labels).to(device)
			##------------------------------------
			## write the predict result below
			##------------------------------------
			output_y = model(x_batch)

			##--------------------------------------------------
			## write code for computing the accuracy below
			## please refer to defined _compute_accuracy() above
			##---------------------------------------------------
			_, y_pred = torch.max(output_y.data, 1)
			accy = _compute_accuracy(y_pred, y_labels)
			acc_list.append(accy)
			f1 = metric(y_pred, y_labels)
			f1_list.append(f1.item())
   
			wandb.log({
       			"test_acc": accy,
				"test_f1" : f1
          	})
	
	acc_mean = np.mean(acc_list)
	acc_std  = np.std(acc_list)
 
	f1_mean = np.mean(f1_list)
	f1_std  = np.std(f1_list)
 
	wandb.log({
		"test_acc_mean": acc_mean,
		"test_acc_std" : acc_std,
		"test_f1_mean" : f1_mean,
		"test_f1_std"  : f1_std
	})
	
			
			
	

if __name__ == '__main__':
	time_start = time.time()
	with wandb.init(project="CSCI646", name=args.wandb_name):
		main()
	time_end = time.time()
	print(f"running time: {(time_end - time_start)/60.0:.3} mins")
	


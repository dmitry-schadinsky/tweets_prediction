import os
import time
import torch
import hparams
from model import Model, save_checkpoint, load_checkpoint
from helper import mode, prepare_datasets, init_training, output, plot_test, test_all
#from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter()
################################################################################################################
#data and model initialisation
params = hparams.Hparams()
init_training(params)
train_loader, test_loader = prepare_datasets(params)
epoch = 1
model = mode(Model(params),params)
optimizer = torch.optim.Adam(model.parameters(), lr = params.lr, eps = 1e-6, weight_decay = params.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 20, eta_min = 5e-6)
################################################################################################################
#load checkpoint
if params.ckpt_pth != '':
	model, optimizer, epoch = load_checkpoint(params.ckpt_pth, model, optimizer)
	epoch += 1 
################################################################################################################
#Training
output("Step 2/2: Training (start from epoch {})".format(epoch), params.log_file)
test_all(test_loader, model, 1, epoch-1)
while (epoch <= params.max_epochs):
	start = time.perf_counter()
	
	model.train()
	total_items = 0
	train_err_epoch = 0.0
	
	output("          training: Epoch {:d} \t lr: {:e}".format(epoch, optimizer.param_groups[0]['lr']), params.log_file)
	count = 0
	for batch in train_loader:
		print("          progress: {:.2f}%".format(count/len(train_loader)*100), end='\r')
		model.zero_grad()
		er, loss, den = model(*model.parse_batch(batch)) #txt_padded, geo_point, txt_lengths = batch
		train_err_epoch += er
		total_items += den
		loss.backward()
		#torch.nn.utils.clip_grad_value_(model.parameters(), params.grad_clip_thresh)
		#torch.nn.utils.clip_grad_norm_(model.parameters(), params.grad_clip_thresh)
		optimizer.step()
		scheduler.step()
		count += 1
	train_err_epoch /= total_items
	
	#writer.add_scalar('Loss/train_per_epoch', train_err_epoch, epoch)
	dur = time.perf_counter()-start
	
	#save checkpoint
	if (epoch % params.epoch_per_save == 0):
		ckpt_pth = os.path.join(params.ckpt_dir, 'ckpt_{}'.format(epoch))
		save_checkpoint(model, optimizer, epoch, ckpt_pth)

	###################################################################################
	#Testing per epoch
	model.eval()
	total_items = 0
	test_err_epoch = 0.0
	for batch in test_loader:
		er, loss, den = model(*model.parse_batch(batch))
		test_err_epoch += er
		total_items += den
	test_err_epoch /= total_items
	#writer.add_scalar('Loss/test_per_epoch', test_err_epoch, epoch)
	output("              train/test error: wav {:e}/{:e} \t Train epoch time: {:6.2f}".format(train_err_epoch, test_err_epoch, dur), params.log_file)
	
	test_all(test_loader, model, 1, epoch)
	epoch += 1
	
output("Training complete", params.log_file)

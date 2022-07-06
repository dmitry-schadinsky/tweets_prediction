import time
import torch
import hparams
from model import Model, load_checkpoint
from helper import mode, prepare_datasets, init_training, output, plot_test, test_all

################################################################################################################
#data and model initialisation
params = hparams.Hparams()
init_training(params)
test_loader = prepare_datasets(params, only_test = True)
################################################################################################################
#load checkpoint
model = mode(Model(params),params)
model, epoch = load_checkpoint(params.ckpt_pth, model)
################################################################################################################
#Training
output("          Testing (epoch {})".format(epoch), params.log_file)
#test_all(test_loader, model, 0, epoch)
test_all(test_loader, model, 1, epoch)

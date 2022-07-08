class Hparams:
	seed = 0

	#paths
	raw_data_dir = '../raw_data/' 
	out_res_dir = 'results/'
	data_dir = out_res_dir + 'data/'
	ckpt_dir = out_res_dir +'ckpt/'
	log_file = data_dir + 'data.log'
	stat_file = data_dir + 'data.stat'
	train_dset_file = data_dir + 'train.data'
	test_dset_file = data_dir + 'test.data'
	res_stat_file = data_dir + 'dists.txt'
	ckpt_pth = ''#ckpt_dir+'ckpt_20' #not empty if training from specific epoch
	
	#for datasets
	train_part = 0.95
	test_part = 1.0 - train_part
	max_dist = 500.0
	n_workers = 6
	pin_mem = True
	part_of_train = 1.0
	part_of_test = 1.0
	
	#data stat (it is calculated and saved in stat_file)
	mean_lat = 0.0
	mean_lon = 0.0
	min_lat = 0.0
	max_lat = 0.0
	min_lon = 0.0
	max_lon = 0.0
	
	#model params
	isize = 256
	emb_dim = 64
	hsize = 512
	lin_size = 1024
	osize = 2000
	init_conc = 50.0
	r = 1.0
	
	#training params	
	max_epochs = 21
	epoch_per_save = 1
	epoch_per_plot = 1
	device = "cuda:0"
	batch_size = 256
	weight_decay = 1e-6
	lr = 1e-3
	grad_clip_thresh = 1.0
	
	#fields for debug statistics on one train and one test files 
	#(filled in dataset.py dset::__init__)
	#(used in helper.py plot_test)
	train_text = ''
	train_lat = 0.0
	train_lon = 0.0
	
	test_text = ''
	test_lat = 0.0
	test_lon = 0.0

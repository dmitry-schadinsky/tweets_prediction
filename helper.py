import os
import glob
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import Dset, collate, get_txt_geo_point
from math import atan, atan2, cos, radians, sin, sqrt, tan, sinh, pi
################################################################################################################
#General

#output on the screen and to log file
def output(string, file_name, need_print = True):
	if need_print:
		print(string)
	log = open(file_name, "a")
	log.write(string+'\n')
	log.close()
	
#move data to computing device (cpu or gpu)
def mode(obj, p, model = False):
	if 'cuda' in p.device:
		obj = obj.to(p.device, non_blocking = p.pin_mem)
	return obj

################################################################################################################
#For working with sphere

#coordinates transformations
def lat_lon_to_sph(lat, lon):
	theta = 0.5*pi-lat
	phi = lon 
	return(theta, phi)
def sph_to_cart(theta, phi, p):
	x = p.r * torch.sin(theta) * torch.cos(phi)
	y = p.r * torch.sin(theta) * torch.sin(phi)
	z = p.r * torch.cos(theta)
	return(x,y,z)
def lat_lon_to_cart(lat, lon, p):
	(theta, phi) = lat_lon_to_sph(lat, lon)
	(x,y,z) = sph_to_cart(theta, phi, p)
	return (x,y,z)

#von mises fisher probability density function
def von_mises_fisher_pdf(x, mu, kappa):
	return kappa/(torch.sinh(kappa)*4*pi) * torch.exp(kappa*(x[0]*mu[0]+x[1]*mu[1]+x[2]*mu[2]))

#distance between two point on sphere
def vincenty_inverse(coords,maxIter=200,tol=10**-12):
	#Constants
	a=6378137.0                             # radius at equator in meters
	f=1/298.257223563                       # flattening of the ellipsoid
	b=(1-f)*a

	phi_1,L_1=coords[0]                       # (lat=L,lon=phi)
	phi_2,L_2=coords[1]

	u_1=atan((1-f)*tan(radians(phi_1)))
	u_2=atan((1-f)*tan(radians(phi_2)))

	L=radians(L_2-L_1)
	Lambda=L                                # set initial value of lambda to L
	

	sin_u1=sin(u_1)
	cos_u1=cos(u_1)
	sin_u2=sin(u_2)
	cos_u2=cos(u_2)

	#Begin iteration
	iters=0
	for i in range(0,maxIter):
		iters+=1

		cos_lambda=cos(Lambda)
		sin_lambda=sin(Lambda)
		sin_sigma=sqrt((cos_u2*sin(Lambda))**2+(cos_u1*sin_u2-sin_u1*cos_u2*cos_lambda)**2)
		cos_sigma=sin_u1*sin_u2+cos_u1*cos_u2*cos_lambda
		sigma=atan2(sin_sigma,cos_sigma)
		if (abs(Lambda) > tol):
			sin_alpha=(cos_u1*cos_u2*sin_lambda)/sin_sigma
		else:
			sin_alpha=cos_u1*cos_u2
		cos_sq_alpha=1-sin_alpha**2
		cos2_sigma_m=cos_sigma-((2*sin_u1*sin_u2)/cos_sq_alpha)
		C=(f/16)*cos_sq_alpha*(4+f*(4-3*cos_sq_alpha))
		Lambda_prev=Lambda
		Lambda=L+(1-C)*f*sin_alpha*(sigma+C*sin_sigma*(cos2_sigma_m+C*cos_sigma*(-1+2*cos2_sigma_m**2)))

		# successful convergence
		diff=abs(Lambda_prev-Lambda)
		if diff<=tol:
			break

	u_sq=cos_sq_alpha*((a**2-b**2)/b**2)
	A=1+(u_sq/16384)*(4096+u_sq*(-768+u_sq*(320-175*u_sq)))
	B=(u_sq/1024)*(256+u_sq*(-128+u_sq*(74-47*u_sq)))
	delta_sig=B*sin_sigma*(cos2_sigma_m+0.25*B*(cos_sigma*(-1+2*cos2_sigma_m**2)-(1/6)*B*cos2_sigma_m*(-3+4*sin_sigma**2)*(-3+4*cos2_sigma_m**2)))

	dist=b*A*(sigma-delta_sig)           # output distance in meters
	dist_km=dist/1000                    # output distance in kilometers
	return dist_km
	
#dist = vincenty_inverse([[39.152501,-84.412977],[39.152505,-84.412946]])

################################################################################################################
#For training

#initialisation and data preparing
def prepare_dir(cur_dir):
	if not os.path.isdir(cur_dir):
		os.makedirs(cur_dir)
		os.chmod(cur_dir, 0o775)
def prepare_dirs(p):
	prepare_dir(p.out_res_dir)
	prepare_dir(p.data_dir)
	prepare_dir(p.ckpt_dir)
def init_training(p):
	torch.manual_seed(p.seed)
	random.seed(p.seed)
	torch.set_num_threads(1)
	if 'cuda' in p.device:
		torch.cuda.set_device(int(p.device.split(':')[1]))
	prepare_dirs(p)
def prepare_dataloaders(dset_file, p, label):
	data_loader = DataLoader(Dset(dset_file,p,label), num_workers = 0, shuffle = True, batch_size = p.batch_size,
				 pin_memory = p.pin_mem, drop_last = True, collate_fn = collate())#p.n_workers
	return data_loader

#datasets preprocessing
def prepare_datasets(p, only_test = False):
	output("Step 1/2: prepare train and test datasets", p.log_file)
	if os.path.isfile(p.stat_file) and os.path.isfile(p.test_dset_file) and os.path.isfile(p.train_dset_file):				
		output("          train and test parts are already saved", p.log_file)
	else:
		files = glob.glob(p.raw_data_dir+'*.csv')
		output("          total number of files {}".format(len(files)), p.log_file)		
		proccesed_files_num = 0
		data_list = []
		min_lat = 90.0
		max_lat = -90.0
		min_lon = 180.0
		max_lon = -180.0
		min_dist = 1.0e10
		max_dist = 0.0
		all_lat = []
		all_lon = []
		for current_file in files:
			with open(current_file, encoding = 'utf-8') as f:
				print("          reading raw data: {:.2f}%".format(proccesed_files_num/len(files)*100), end='\r')
				for line in f:
					parts = line.strip().split(';')
					try:
						cur_geo = [float(x) for x in parts[15][1:-1].split(',')]#[lon1, lat1, lon2, lat2]
					except:
						continue

					cur_geo = [[cur_geo[1],cur_geo[0]],[cur_geo[3],cur_geo[2]]]
					dist = vincenty_inverse(cur_geo)
					if dist > p.max_dist:
						continue
					data_list += [[parts[3],cur_geo]]
					
					#stat
					min_lat = min(min_lat,cur_geo[0][0])
					min_lon = min(min_lon,cur_geo[0][1])
					max_lat = max(max_lat,cur_geo[1][0])
					max_lon = max(max_lon,cur_geo[1][1])
					all_lat += [0.5*(cur_geo[0][0]+cur_geo[1][0])]
					all_lon += [0.5*(cur_geo[0][1]+cur_geo[1][1])]
					min_dist = min(min_dist, dist)
					max_dist = max(max_dist, dist)
					
			proccesed_files_num += 1
		
		#saving stat
		output("min, max lat: {}, {}".format(min_lat, max_lat), p.stat_file, False)
		output("min, max lon: {}, {}".format(min_lon, max_lon), p.stat_file, False)
		output("min, max dist: {}, {}".format(min_dist, max_dist), p.stat_file, False)
		output('Mean lat: {}'.format(sum(all_lat)/len(all_lat)), p.stat_file, False)
		output('Mean lon: {}'.format(sum(all_lon)/len(all_lon)), p.stat_file, False)
		
		output("          creating and saving train and test parts", p.log_file)		
		new_len = max(int(len(data_list)*p.test_part),1)
		random.shuffle(data_list)
		f_test = open(p.test_dset_file, 'w')
		f_train = open(p.train_dset_file, 'w')
		for i, item in enumerate(data_list):
			print("        progress: {:.2f}%".format(i/len(data_list)*100), end='\r')
			if i < new_len:	
				f_test.write("{}; {}; {}\n".format(item[0], 0.5*(item[1][0][0]+item[1][1][0]), 0.5*(item[1][0][1]+item[1][1][1])))
			else:
				if only_test:
					break
				f_train.write("{}; {}; {}\n".format(item[0], 0.5*(item[1][0][0]+item[1][1][0]), 0.5*(item[1][0][1]+item[1][1][1])))
		f_test.close()
		f_train.close()
		output("          train and test parts successfully saved", p.log_file)
		
	with open(p.stat_file, encoding = 'utf-8') as f:
			for line in f:
				parts = line.strip().split(':')
				if parts[0] == 'min, max lat':
					parts = parts[1].strip().split(',')
					p.min_lat = float(parts[0])
					p.max_lat = float(parts[1])
				elif parts[0] == 'min, max lon':
					parts = parts[1].strip().split(',')
					p.min_lon = float(parts[0])
					p.max_lon = float(parts[1])
				elif parts[0] == 'Mean lat':
					p.mean_lat = float(parts[1])
				elif parts[0] == 'Mean lon':
					p.mean_lon = float(parts[1])
	
	if only_test:
		output("          preparing test dataloader", p.log_file)
		test_loader = prepare_dataloaders(p.test_dset_file,p,'test')
		output("          test dataloaders are ready", p.log_file)
		return test_loader
	
	
	output("          preparing train and test dataloaders", p.log_file)
	train_loader = prepare_dataloaders(p.train_dset_file,p,'train')
	test_loader = prepare_dataloaders(p.test_dset_file,p,'test')
	output("          train and test dataloaders are ready", p.log_file)
	return train_loader, test_loader

################################################################################################################
#For testing

#results for one train and one test file
def plot_test(model, epoch, label='train'):
	model.eval()
	file = model.param.out_res_dir+label+"_prod_{}.png".format(epoch)
	file1 = model.param.out_res_dir+label+"_sum_{}.png".format(epoch)
	if label == 'train':
		txt, geo_point = get_txt_geo_point([model.param.train_text, model.param.train_lat, model.param.train_lon],model.param)
	else:
		txt, geo_point = get_txt_geo_point([model.param.test_text, model.param.test_lat, model.param.test_lon],model.param)
		
	lat_pred, lon_pred, lat_sum_pred, lon_sum_pred = model.infer_debug(mode(txt.unsqueeze(-1),model.param), mode(geo_point.unsqueeze(0),model.param), file, file1)
	lat_pred, lon_pred = degrees(lat_pred)+model.param.mean_lat, degrees(lon_pred)+model.param.mean_lon
	lat_sum_pred, lon_sum_pred = degrees(lat_sum_pred)+model.param.mean_lat, degrees(lon_sum_pred)+model.param.mean_lon
	lat, lon = degrees(geo_point[0].item())+model.param.mean_lat, degrees(geo_point[1].item())+model.param.mean_lon
	dist1 = vincenty_inverse([[lat, lon],[lat_pred, lon_pred]])
	dist2 = vincenty_inverse([[lat, lon],[lat_sum_pred, lon_sum_pred]])
	output("              "+label+" file error distance1 = {:6.2f} km".format(dist1), model.param.log_file)
	output("              "+label+" file error distance2 = {:6.2f} km".format(dist2), model.param.log_file)
	output("              {},{},{},{}".format(lat, lon, lat_pred, lon_pred), model.param.log_file)

#collect statistics for all test data
def test_all(test_loader, model, mod, epoch):
	model.eval()	
	all_dist = []
	i = 0
	for batch in test_loader:
		print("        test_all progress: {:.2f}%".format(i/len(test_loader)*100), end='\r')
		txt_padded, geo_point, txt_lengths = model.parse_batch(batch)
		dist = model.infer(txt_padded, geo_point, mod)
		all_dist += [dist]
		i += 1

	all_dist = torch.cat(all_dist)
	output("              Average dist error = {}".format(all_dist.mean()), model.param.log_file)
	output("              Quantile 0.9 = {}".format(all_dist.quantile(0.9,0,False,interpolation='lower')), model.param.log_file)
	output("              Quantile 0.75 = {}".format(all_dist.quantile(0.75,0,False,interpolation='lower')), model.param.log_file)
	output("              Quantile 0.5 = {}\n".format(all_dist.quantile(0.5,0,False,interpolation='lower')), model.param.log_file)
	
	
	all_dist_np = np.array(all_dist.tolist())
	np.savetxt(model.param.res_stat_file, all_dist_np)
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title('Histogram of all test distances errors')
	ax.set_xlabel('km')
	ax.set_ylabel('percent')
	ax.grid()
	ax.hist(all_dist_np, bins = list(range(0,int(all_dist.max()),100)), rwidth = 0.8, density = True)
	plt.savefig(model.param.out_res_dir+"hist_{}.png".format(epoch))

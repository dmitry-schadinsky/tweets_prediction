import torch
from math import pi, radians, degrees
from helper import mode, lat_lon_to_cart, von_mises_fisher_pdf, sph_to_cart, vincenty_inverse
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt

class Model(nn.Module):
	def __init__(self, p):
		super().__init__()
		self.param = p
		self.embedding = nn.Embedding(self.param.isize, self.param.emb_dim)
		
		hsize = self.param.hsize
		
		self.conv1 = nn.Conv1d(in_channels = self.param.emb_dim, out_channels = hsize, kernel_size = 7, stride=1, padding=3)
		self.act1 = nn.ReLU()
		self.pool1 = nn.MaxPool1d(kernel_size = 3, stride=None, padding=0, dilation=1)
		self.bn1 = nn.BatchNorm1d(hsize)

		self.conv2 = nn.Conv1d(in_channels = hsize, out_channels = hsize, kernel_size = 7, stride=1, padding=3)
		self.act2 = self.act1
		self.pool2 = nn.MaxPool1d(kernel_size = 3, stride=None, padding=0, dilation=1)
		self.bn2 = nn.BatchNorm1d(hsize)
		
		self.conv3 = nn.Conv1d(in_channels = hsize, out_channels = hsize, kernel_size = 3, stride=1, padding=1)
		self.act3 = self.act1
		self.bn3 = nn.BatchNorm1d(hsize)
		
		self.conv4 = nn.Conv1d(in_channels = hsize, out_channels = hsize, kernel_size = 3, stride=1, padding=1)
		self.act4 = self.act1
		self.bn4 = nn.BatchNorm1d(hsize)
		
		self.conv5 = nn.Conv1d(in_channels = hsize, out_channels = hsize, kernel_size = 3, stride=1, padding=1)
		self.act5 = self.act1
		self.bn5 = nn.BatchNorm1d(hsize)
		
		self.conv6 = nn.Conv1d(in_channels = hsize, out_channels = hsize, kernel_size = 3, stride=1, padding=1)
		self.act6 = self.act1
		self.pool6 = nn.MaxPool1d(kernel_size = 3, stride=None, padding=0, dilation=1)
		self.bn6 = nn.BatchNorm1d(hsize)
		
		self.linear7 = nn.Linear(hsize, self.param.lin_size)
		self.act7 = self.act1
		self.linear8 = nn.Linear(self.param.lin_size, self.param.osize)
				
		#calc mean values for self.mean_direction initialization
		min_lat = radians(self.param.min_lat-self.param.mean_lat)
		max_lat = radians(self.param.max_lat-self.param.mean_lat)
		min_lon = radians(self.param.min_lon-self.param.mean_lon)
		max_lon = radians(self.param.max_lon-self.param.mean_lon)
		mean_direction = torch.rand(self.param.osize, 2, dtype=torch.float)
		mean_direction[:,0] *= (max_lat-min_lat)
		mean_direction[:,0] += min_lat
		mean_direction[:,1] *= (max_lon-min_lon)
		mean_direction[:,1] += min_lon
		
		self.mean_direction = nn.Parameter(mean_direction)
		self.concentration = nn.Parameter(torch.full([self.param.osize], self.param.init_conc, dtype=torch.float))
		
	def network(self, txt_padded):
		w = self.embedding(txt_padded)
		
		w = self.conv1(w.permute(1,2,0))
		w = self.act1(w)
		w = self.pool1(w)
		w = self.bn1(w)
		
		w = self.conv2(w)
		w = self.act2(w)
		w = self.pool2(w)
		w = self.bn2(w)

		w = self.conv3(w)
		w = self.act3(w)
		w = self.bn3(w)

		w = self.conv4(w)
		w = self.act4(w)
		w = self.bn4(w)

		w = self.conv5(w)
		w = self.act5(w)
		w = self.bn5(w)

		w = self.conv6(w)
		w = self.act6(w)
		w = self.pool6(w)
		w = self.bn6(w)

		w = w.sum(2)/w.shape[2]
		w = self.linear7(w)
		w = self.act7(w)
		w = self.linear8(w)
		w = w.softmax(-1)#B, osize
		
		return w
	
	def forward(self, txt_padded, geo_point, txt_lengths):
		w = self.network(txt_padded)		
		
		x,y,z = lat_lon_to_cart(geo_point[:,0], geo_point[:,1], self.param)#B
		mu_x,mu_y,mu_z = lat_lon_to_cart(self.mean_direction[:,0], self.mean_direction[:,1], self.param)
		loss = w * von_mises_fisher_pdf([x.unsqueeze(1),y.unsqueeze(1),z.unsqueeze(1)],
		 								[mu_x.unsqueeze(0),mu_y.unsqueeze(0),mu_z.unsqueeze(0)], 
		 								self.concentration)
		loss = loss.sum(-1)
		loss = (-(loss.log())).sum()
		er = loss.item()
		den = txt_lengths.shape[0]
		return er, loss, den

	def parse_batch(self, batch):
		txt_padded, geo_point, txt_lengths = batch
		txt_padded = mode(txt_padded, self.param)
		geo_point = mode(geo_point, self.param)
		return txt_padded, geo_point, txt_lengths
	
	def infer_debug(self, txt_padded, geo_point, file, file1):
		w = self.network(txt_padded)				
		w = w[0]#osize
		
		theta, phi, W, lat, lon = calc_geo_point(w, self, 0)
		#plot_on_sphere(theta, phi, W, self, file)
		theta, phi, W, lat_sum, lon_sum = calc_geo_point(w, self, 1)
		#plot_on_sphere(theta, phi, W, self, file1)
		
		return lat, lon, lat_sum, lon_sum
	
	def infer(self, txt_padded, geo_point, mod):		
		w = self.network(txt_padded)
		dist = torch.zeros(w.shape[0], dtype = torch.float)
		for i in range(w.shape[0]):
			theta, phi, W, lat_pred, lon_pred = calc_geo_point(w[i,:], self, mod)
			lat_pred, lon_pred = degrees(lat_pred)+self.param.mean_lat, degrees(lon_pred)+self.param.mean_lon
			lat, lon = degrees(geo_point[i][0].item())+self.param.mean_lat, degrees(geo_point[i][1].item())+self.param.mean_lon
			dist[i] = vincenty_inverse([[lat, lon],[lat_pred, lon_pred]])
		return dist
		
def von_mises_mixture(theta, phi, w, model):
	x,y,z = sph_to_cart(theta, phi, model.param)
	mu_x,mu_y,mu_z = lat_lon_to_cart(model.mean_direction[:,0], model.mean_direction[:,1], model.param)
	val =  von_mises_fisher_pdf([x.unsqueeze(1),y.unsqueeze(1),z.unsqueeze(1)],
								[mu_x.unsqueeze(0),mu_y.unsqueeze(0),mu_z.unsqueeze(0)], 
								model.concentration) ** w
	val = val.prod(-1)
	return val

def von_mises_mixture_sum(theta, phi, w, model):
	x,y,z = sph_to_cart(theta, phi, model.param)
	mu_x,mu_y,mu_z = lat_lon_to_cart(model.mean_direction[:,0], model.mean_direction[:,1], model.param)
	val =  von_mises_fisher_pdf([x.unsqueeze(1),y.unsqueeze(1),z.unsqueeze(1)],
	 							[mu_x.unsqueeze(0),mu_y.unsqueeze(0),mu_z.unsqueeze(0)], 
	 							model.concentration) * w
	val = val.sum(-1)
	return val

def calc_geo_point(w, model, mod = 0):
	theta, phi = torch.linspace(0, pi, 180), torch.linspace(-pi, pi, 2*180)
	theta, phi = torch.meshgrid(theta, phi, indexing='xy')
	old_shape0 = theta.shape[0]
	old_shape1 = theta.shape[1]
	theta = mode(theta.reshape(-1),model.param)
	phi = mode(phi.reshape(-1),model.param)
	if mod == 0:
		val = von_mises_mixture(theta, phi, w, model)
	else:
		val = von_mises_mixture_sum(theta, phi, w, model)
	ind = val.argmax()
	lat_predict = (0.5*pi-theta[ind]).item()
	lon_predict = phi[ind].item()

	theta = theta.detach().cpu().reshape(old_shape0,old_shape1)
	phi = phi.detach().cpu().reshape(old_shape0,old_shape1)
	val = val.detach().cpu().reshape(old_shape0,old_shape1)
	return theta, phi, val, lat_predict, lon_predict

def plot_on_sphere(theta, phi, W, model, file):
	x,y,z = sph_to_cart(theta, phi, model.param)

	color_dimension = W.numpy() # change to desired fourth dimension
	minn, maxx = color_dimension.min(), color_dimension.max()
	norm = matplotlib.colors.Normalize(minn, maxx)
	m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
	m.set_array([])
	fcolors = m.to_rgba(color_dimension)

	fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
	plt.xlabel("X")
	plt.ylabel("Y")
	surf = ax.plot_surface(x.numpy(),y.numpy(),z.numpy(), rstride=1, cstride=1, facecolors=fcolors, vmin=minn, vmax=maxx, shade=False, alpha=0.7)
	fig.colorbar(m)
	ax.view_init(0,0)
	plt.savefig(file)
	plt.show()

def save_checkpoint(model, optimizer, epoch, ckpt_pth):
	torch.save({'model': model.state_dict(),
		    'optimizer': optimizer.state_dict(),
		    'epoch': epoch}, ckpt_pth)

def load_checkpoint(ckpt_pth, model, optimizer=None):
	ckpt_dict = torch.load(ckpt_pth,map_location=model.param.device)
	model.load_state_dict(ckpt_dict['model'])
	epoch = ckpt_dict['epoch']
	if optimizer is None:
		return model, epoch
	optimizer.load_state_dict(ckpt_dict['optimizer'])

	return model, optimizer, epoch

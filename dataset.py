import glob
import torch
import random
from math import radians
from torch.utils.data import Dataset

def output(string, file_name, need_print = True):
	if need_print:
		print(string)
	log = open(file_name, "a")
	log.write(string+'\n')
	log.close()

class Dset(Dataset):
	def __init__(self, current_file, param, label):
		self.p = param
		part = 1.0
		if label == 'train':
			part = param.part_of_train
		elif label == 'test':
			part = param.part_of_test
		self.data_list = []
		with open(current_file, encoding = 'utf-8') as f:
			for line in f:
				parts = line.strip().split(';')
				self.data_list += [[parts[0],float(parts[1]),float(parts[2])]]
		if part < 1.0:
			new_len = max(int(len(self.data_list)*part),1)
			self.data_list = random.sample(self.data_list, new_len)
		if label == 'train':
			param.train_text = self.data_list[0][0]
			param.train_lat = self.data_list[0][1]
			param.train_lon = self.data_list[0][2]
		elif label == 'test':
			param.test_text = self.data_list[2][0]
			param.test_lat = self.data_list[2][1]
			param.test_lon = self.data_list[2][2]
		
		output("Data size : {}".format(len(self.data_list)),self.p.log_file)
	
	def __getitem__(self, index):
		txt, geo_point =  get_txt_geo_point(self.data_list[index], self.p)
		return txt, geo_point

	def __len__(self):
		return len(self.data_list)

def get_txt_geo_point(data_item, p):
	txt = data_item[0]
	txt = torch.IntTensor(list(txt.encode('utf8')))
	geo_point = torch.FloatTensor([radians(data_item[1]-p.mean_lat), radians(data_item[2]-p.mean_lon)])
	return (txt, geo_point)
	
class collate():
	def __call__(self, batch):
		txt_lengths = torch.IntTensor([x[0].shape[0] for x in batch])
		txt_padded = torch.zeros(torch.max(txt_lengths), len(batch), dtype = torch.long) #Frames,Batch
		geo_point = torch.zeros(len(batch), 2, dtype = torch.float)		
		for i in range(len(batch)):
			txt = batch[i][0]
			txt_padded[:txt.shape[0], i] = txt
			geo_point[i,:] = batch[i][1]
			
		return txt_padded, geo_point, txt_lengths

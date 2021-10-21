# Copyright 2021 [name of copyright owner]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import cv2, h5py
import os, glob, random
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import gc

SEED = 32
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
os.environ['PYTHONHASHSEED']=str(SEED)

class NRDataset(Dataset):
	def __init__(self, data_path, grayScale = True, batch_size = 8, chunk_size = 10, passes=10, max_dim = 720, geo = False):
		self.cnt = 0
		self.chunk_cnt = 0
		self.done = 0
		self.batch_size = batch_size
		self.chunk_size = chunk_size
		self.passes = passes
		self.chunk_data = None
		self.grayScale = grayScale
		self.max_dim = max_dim

		self.data = h5py.File(data_path, "r")

		self.geo = True if 'geopatch' in self.data.keys() and geo else False

		self.names = list(self.data['imgs'].keys())
		self.names = list(set(['{:s}__{:s}'.format(*n.split('__')[:2]) for n in self.names]))
		random.shuffle(self.names)
		self.regularize_names()
		round_sz = len(self.names) // (batch_size * chunk_size)
		self.names = self.names[:round_sz * batch_size * chunk_size] # trim dataset size to a multiple of batch&chunk  

		self.toTensor = transforms.ToTensor()
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.load_chunk()


	def __len__(self):
		return len(self.names) // (self.chunk_size * self.batch_size)

	def load_chunk(self):
		print('loading chunk [{:d}]...'.format(self.chunk_cnt), end='', flush=True)
		self.chunk_data = [] ; gc.collect()
		N = len(self.names) // self.chunk_size
		for i in range(0, N , self.batch_size):
			batch = []
			for b in range(self.batch_size):
				#Read the data from disk
				idx = N * self.chunk_cnt + i + b
				key = self.names[idx]
				X =self.data['imgs/'+key+'__1'][...]
				Y =self.data['imgs/'+key+'__2'][...]
				x_kps = self.data['kps/'+key+'__1'][...]
				y_kps = self.data['kps/'+key+'__2'][...]

				if self.geo: #grab GeoPatches
					x_geo = self.data['geopatch/'+key+'__1'][...]
					y_geo = self.data['geopatch/'+key+'__2'][...]
					#permute axis to (N,1,H,W)
					x_geo = np.transpose(x_geo, (2,0,1))[:,np.newaxis,:,:].astype(np.float32)/255.
					y_geo = np.transpose(y_geo, (2,0,1))[:,np.newaxis,:,:].astype(np.float32)/255.

				# Resize image and keypoint attributes if the image is too large
				r_x = max(X.shape[:2])/self.max_dim
				r_y = max(Y.shape[:2])/self.max_dim
				if r_x > 1.0:
					X = cv2.resize(X, None, fx = 1/r_x, fy = 1/r_x)
					x_kps[:,:2]/= r_x ; x_kps[:,3]/= r_x
				if r_y > 1.0:
					Y = cv2.resize(Y, None, fx = 1/r_y, fy = 1/r_y)
					y_kps[:,:2]/= r_y ; x_kps[:,3]/= r_y				
	
				X = self.toTensor(X) ; Y = self.toTensor(Y)

				if self.grayScale:
					X = torch.mean(X,0,True) ; Y = torch.mean(Y,0,True)

				batch.append((X, x_kps,  Y, y_kps) if not self.geo else (X, x_kps,x_geo,  Y, y_kps, y_geo))
			self.chunk_data.append(batch)
		
		self.chunk_cnt+=1
		self.done=0
		if self.chunk_cnt == self.chunk_size:
			self.chunk_cnt = 0


		print('done.')


	def get_raw_batch(self, idx):
		batch = list(zip(*self.chunk_data[idx]))
		return batch

	def regularize_names(self):
		dk = {}
		new_names = []
		from collections import deque

		for n in self.names:
			key = n.split('__')[0]
			if key in dk: dk[key].append(n)
			else: dk[key] = deque([n])

		#for v in dk.values():
		#	print(len(v))
				
		done = False
		while not done:
			cnt=0
			for k,v in dk.items():
				if len(dk[k])==0:
					cnt+=1
				else:
					new_names.append(dk[k].pop())
			if cnt==len(dk):
				done = True

		self.names = new_names

	def __getitem__(self, idx):	
		if self.done == self.passes:
			self.load_chunk()

		batch = list(zip(*self.chunk_data[idx]))
		batch = self.prepare_batch(batch)
		return batch

	def __iter__(self):
		self.cnt = 0
		return self

	def __next__(self):
		if self.cnt == self.__len__():
			self.done+=1
			self.cnt=0
			raise StopIteration
		else:
			self.cnt+=1
			return self.__getitem__(self.cnt-1)

	def prepare_batch(self, batch, max_kps = 128):
		dev = self.device

		if not self.geo:
			X, x_kps, Y, y_kps = batch
		else:
			X, x_kps, x_geo, Y, y_kps, y_geo = batch

		sampled_x_kps, sampled_y_kps = [], []
		sampled_x_geo, sampled_y_geo = [], []

		for b in range(len(x_kps)):
			rnd_idx = np.arange(len(x_kps[b]))
			np.random.shuffle(rnd_idx)
			rnd_idx = rnd_idx[:max_kps]
			sampled_x_kps.append(x_kps[b][rnd_idx])
			sampled_y_kps.append(y_kps[b][rnd_idx])
			if self.geo:
				sampled_x_geo.append(x_geo[b][rnd_idx])
				sampled_y_geo.append(y_geo[b][rnd_idx])

		x_kps = sampled_x_kps
		y_kps = sampled_y_kps
		if self.geo:
			x_geo = sampled_x_geo
			y_geo = sampled_y_geo

		nkps_x, nkps_y = [], []
		nori_x, nori_y, nscale_x, nscale_y = [], [], [], []
		ngeo_x, ngeo_y = [], []
		idx_keys = []
		Hs_x, Ws_x = [], []
		Hs_y, Ws_y = [], []

		X_dict, Y_dict = {}, {}

		for b in range(len(x_kps)): #for each image in the batch, prepare keypoints
			H, W = X[b].shape[-2:]
			imgSize = np.array([W-1,H-1], dtype = np.float32)
			nkp_x = x_kps[b][:,:2] / imgSize * 2 - 1
			Hs_x += [H] * len(x_kps[b])
			Ws_x += [W] * len(x_kps[b]) 

			H, W = Y[b].shape[-2:]
			imgSize = np.array([W-1,H-1], dtype = np.float32)
			nkp_y = y_kps[b][:,:2] / imgSize * 2 - 1
			Hs_y += [H] * len(y_kps[b]) 
			Ws_y += [W] * len(y_kps[b])

			idx = [b] * len(x_kps[b])
			idx_keys+=idx
			nkps_x.append(nkp_x); nkps_y.append(nkp_y)
			nori_x.append(x_kps[b][:,2]); nori_y.append(y_kps[b][:,2])
			nscale_x.append(x_kps[b][:,3]); nscale_y.append(y_kps[b][:,3])
			
			if self.geo:
				ngeo_x.append(x_geo[b]) ; ngeo_y.append(y_geo[b])
			

			X_dict[b] = {'img':X[b].to(dev)}
			Y_dict[b] = {'img':Y[b].to(dev)}
			
		nkps_x = np.vstack(nkps_x); nkps_y = np.vstack(nkps_y)
		scale_x = np.concatenate(nscale_x); scale_y = np.concatenate(nscale_y)
		ori_x = np.concatenate(nori_x); ori_y = np.concatenate(nori_y)
		if self.geo:
			geo_x = np.concatenate(ngeo_x) ; geo_y = np.concatenate(ngeo_y)
			geo_x = torch.from_numpy(geo_x).to(dev)
			geo_y = torch.from_numpy(geo_y).to(dev)
		
		#scale_x = np.ones_like(scale_x)*2. ; scale_y = np.ones_like(scale_y)*2.
		ori_x = np.zeros_like(scale_x) ; ori_y = np.zeros_like(scale_y)


		Hs_x, Ws_x = np.array(Hs_x, dtype = np.float32), np.array(Ws_x, dtype = np.float32)
		Hs_y, Ws_y = np.array(Hs_y, dtype = np.float32), np.array(Ws_y, dtype = np.float32)

		theta_x =   [torch.from_numpy(nkps_x).to(dev), 
								torch.from_numpy(scale_x).to(dev),
								torch.from_numpy(ori_x).to(dev),
								torch.from_numpy(Hs_x).to(dev),
								torch.from_numpy(Ws_x).to(dev)]

		theta_y =  [torch.from_numpy(nkps_y).to(dev), 
								torch.from_numpy(scale_y).to(dev),
								torch.from_numpy(ori_y).to(dev),
								torch.from_numpy(Hs_y).to(dev),
								torch.from_numpy(Ws_y).to(dev)]

		if not self.geo:
			return X_dict, Y_dict, theta_x, theta_y , idx_keys
		else:
			return X_dict, Y_dict, theta_x, theta_y , geo_x, geo_y, idx_keys
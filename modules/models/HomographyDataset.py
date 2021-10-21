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


import torch
from torch import nn
from torch.utils.data import Dataset
import torch.utils.data as data
from torchvision import transforms
import torch.nn.functional as F

import cv2
import kornia

import kornia.augmentation as K
import glob
import random

import numpy as np
import pdb

random.seed(0)
torch.manual_seed(0)

def generateRandomHomography(shape, GLOBAL_MULTIPLIER = 0.35):

	theta = np.radians(np.random.normal(0, 2.0*GLOBAL_MULTIPLIER)) 

	scale = np.random.normal(1.0, 0.15*GLOBAL_MULTIPLIER)
	if scale < 1.0: # get the right part of the gaussian
		scale += 2*(1.0 - scale)
		scale = 1.0/scale

	tx , ty = -shape[1]/2.0, -shape[0]/2.0
	txn, tyn = np.random.normal(0, 1.0*GLOBAL_MULTIPLIER, 2) #translation error
	c, s = np.cos(theta), np.sin(theta)

	sx , sy = np.random.normal(0,0.04*GLOBAL_MULTIPLIER,2)
	p1 , p2 = np.random.normal(0,0.005*GLOBAL_MULTIPLIER,2)


	H_t = np.array(((1,0, tx), (0, 1, ty), (0,0,1))) #t

	H_r = np.array(((c,-s, 0), (s, c, 0), (0,0,1))) #rotation,
	H_a = np.array(((1,sy, 0), (sx, 1, 0), (0,0,1))) # affine
	H_p = np.array(((1, 0, 0), (0 , 1, 0), (p1,p2,1))) # projective

	H_s = np.array(((scale,0, 0), (0, scale, 0), (0,0,1))) #scale
	H_b = np.array(((1.0,0,-tx +txn), (0, 1, -ty + tyn), (0,0,1))) #t_back,

	#H = H_e * H_s * H_a * H_p
	H = np.dot(np.dot(np.dot(np.dot(np.dot(H_b,H_s),H_p),H_a),H_r),H_t)

	return H

class AugmentationPipe(nn.Module):
	def __init__(self, device) -> None:
		super(AugmentationPipe, self).__init__()
		self.half = 16
		self.device = device
		self.ORB = cv2.ORB_create(nfeatures = 2048, scaleFactor = 1.6, nlevels = 4)

	def rnd_kps(self, shape, n = 256):
		h, w = shape
		kps = torch.rand(size = (3,n)).to(self.device)
		kps[0,:]*=w
		kps[1,:]*=h
		kps[2,:] = 1.0

	#check with a regular grid
	#  x = np.linspace(400,600,50)
	#  y = np.linspace(300,500,50)
	#  gridx, gridy = np.meshgrid(x,y)
	#  xy = np.dstack((gridx[..., np.newaxis],gridy[..., np.newaxis]))
	#  xy = xy.reshape(-1,2).T
	#  kps = np.ones((3,xy.shape[-1]), dtype=np.float32)
	#  kps[:-1,:] = xy
	#  kps = torch.tensor(kps, dtype = torch.float32)

		return kps

	def orb_kps(self, img, n = 128):
		cv_img = (img*255.0).permute(1,2,0).flip(2).cpu().numpy().astype(np.uint8)
		kps = self.ORB.detect(cv_img, None)
		#print('DETECTED ', len(kps), ' Keypoints')

		random.shuffle(kps)
		coords = [[kp.pt[0], kp.pt[1], 1.0] for kp in kps[:n]] # select n random keypoints
		return torch.from_numpy( np.array(coords, dtype=np.float32).T )

	def forward(self, input):

		valid_kps, valid_warped = [], []
		output = []

		for i in range(len(input)):
			input[i] = input[i].to(self.device)
			shape = input[i].shape[-2:]
			h,w = shape

			H = torch.tensor([generateRandomHomography(shape)], dtype = torch.float32).to(self.device)

			output.append( kornia.transform.warp_perspective(input[i].unsqueeze(0), H, dsize = shape, border_mode = 'border').squeeze(0) )

			kps = self.orb_kps(input[i])

			warped = torch.matmul(H[0,...].cpu(),kps.cpu())
			warped = warped / warped[2,...] 

			#remove keypoints lying outside 
			invalids = np.where(warped[0,:] < 0)[0].tolist() + np.where(warped[0,:] >= w)[0].tolist()
			invalids += np.where(warped[1,:] < 0)[0].tolist() + np.where(warped[1,:] >= h)[0].tolist()
			valids = list (set(range(warped.shape[1])) - set(invalids))

			valid_kps.append(kps[:-1, valids].numpy().T)
			valid_warped.append(warped[:-1, valids].numpy().T)

		return input, valid_kps, output, valid_warped

class Phototour(Dataset):
	def __init__(self, images_path,  n=32, max_img = 1000, max_dim = 1000, grayScale = True, batch_size = 8):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.n = int(np.round(n / batch_size) * batch_size)
		self.batch_size = batch_size
		toTensor = transforms.ToTensor()
		self.cv_imtype = 0 if grayScale else 1
		self.fast = cv2.FastFeatureDetector_create(25)
		self.warper = AugmentationPipe(device = self.device).to(self.device)
		self.cnt = 0

		self.images = []
		for p in (glob.glob(images_path + '/*.jpg') + glob.glob(images_path + '/*.png'))[:max_img]:
			img = cv2.imread(p, self.cv_imtype)
			n_kps = len(self.fast.detect(img, None))
			if n_kps < 100:
				continue
			largest_dim = max(img.shape[:2])
			#print(largest_dim)
			r = largest_dim / max_dim
			
			if r != 1.0:
				img = cv2.resize(img, None, fx = 1/r, fy = 1/r)
			# nx, ny =  np.random.randint(300,1000,2)
			# img = cv2.resize(img,(nx, ny))
			self.images.append(toTensor(img))

	def __len__(self):
		return self.n // self.batch_size

	def get_raw_batch(self, idx):
		batch = [self.images[(idx*self.batch_size + j)%len(self.images)].to(self.device) for j in range(self.batch_size)]
		return self.warper(batch)
	
	def __getitem__(self, idx):
		batch = [self.images[(idx*self.batch_size + j)%len(self.images)].to(self.device) for j in range(self.batch_size)]
		X_dict, Y_dict, theta_x, theta_y , idx_keys = self.prepare_batch(batch)
		return (X_dict, Y_dict, theta_x, theta_y , idx_keys)

	def __iter__(self):
		self.cnt=0
		return self
			
	def __next__(self):
		if self.cnt == self.__len__():
			self.cnt=0
			raise StopIteration
		else:
			self.cnt+=1
			return self.__getitem__(self.cnt-1)

	def prepare_batch(self, batch):
		dev = self.device
		Y, y_kps, X, x_kps = self.warper(batch) #Apply random homography warp and generate some correspondences

		nkps_x, nkps_y = [], []
		idx_keys = []
		Hs_x, Ws_x = [], []
		Hs_y, Ws_y = [], []

		X_dict, Y_dict = {}, {}

		for b in range(len(x_kps)): #for each image in the batch, prepare keypoints
			H, W = X[b].shape[-2:]
			imgSize = np.array([W-1,H-1], dtype = np.float32)
			nkp_x = x_kps[b] / imgSize * 2 - 1
			Hs_x += [H] * len(x_kps[b])
			Ws_x += [W] * len(x_kps[b]) 

			H, W = Y[b].shape[-2:]
			imgSize = np.array([W-1,H-1], dtype = np.float32)
			nkp_y = y_kps[b] / imgSize * 2 - 1
			Hs_y += [H] * len(y_kps[b]) 
			Ws_y += [W] * len(y_kps[b])

			idx = [b] * len(x_kps[b])
			idx_keys+=idx
			nkps_x.append(nkp_x); nkps_y.append(nkp_y)

			X_dict[b] = {'img':X[b]}
			Y_dict[b] = {'img':Y[b]}
			
		nkps_x = np.vstack(nkps_x); nkps_y = np.vstack(nkps_y)

		N = len(nkps_x)
		ori_x, ori_y = np.zeros(N, dtype = np.float32), np.zeros(N, dtype = np.float32)
		scale_x, scale_y = np.full(N, 1, dtype = np.float32), np.full(N, 1, dtype = np.float32)
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

		return X_dict, Y_dict, theta_x, theta_y , idx_keys

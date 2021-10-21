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


import torch, kornia
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp


def gaussian(window_size, sigma):
	gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
	return gauss/gauss.sum()

def create_window(window_size, channel):
	_1D_window = gaussian(window_size, 1.5).unsqueeze(1)
	_2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
	window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
	return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
	mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
	mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)
  
	mu1_sq = mu1.pow(2)
	mu2_sq = mu2.pow(2)
	mu1_mu2 = mu1*mu2

	sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
	sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
	sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

	C1 = 0.01**2
	C2 = 0.03**2

	ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

	if size_average:     
		return ssim_map.mean()
	else:
		return ssim_map.mean(1).mean(1).mean(1)

def SSIM_loss(img1, img2, window_size = 5, size_average = True):
	(_, channel, _, _) = img1.size()
	window = create_window(window_size, channel)
	
	if img1.is_cuda:
		window = window.cuda(img1.get_device())
	window = window.type_as(img1)
	
	return 1.0 - _ssim(img1, img2, window, window_size, channel, size_average)


def sharpness_loss(img):
	# dx = img[:, :, :, 1:] - img[:, :, :, :-1]
	# dy = img[:, :, 1:, :] - img[:, :, :-1, :]

	sharpness = kornia.laplacian(img,3).std()
	zero_tensor = torch.tensor([0.]).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
	return torch.max(zero_tensor, 0.1 - sharpness)*6.0


def hardnet_loss(X,Y, margin = 0.5):

	if X.size() != Y.size() or X.dim() != 2 or Y.dim() != 2:
		raise RuntimeError('Error: X and Y shapes must match and be 2D matrices')

	dist_mat = torch.sqrt( 2.*(1.-torch.mm(X,Y.t())) )
	dist_pos = torch.diag(dist_mat)
	dist_neg = dist_mat + 100.*torch.eye(*dist_mat.size(), dtype = dist_mat.dtype, 
					device = dist_mat.get_device() if dist_mat.is_cuda else torch.device("cpu"))

	#filter repeated patches on negative distances to avoid weird stuff on gradients
	dist_neg = dist_neg + dist_neg.le(0.008).float()*100.

	hard_neg = torch.min(dist_neg, 1)[0]

	triplet_loss = torch.clamp(margin + dist_pos - hard_neg, min=0.)

	return triplet_loss.mean()
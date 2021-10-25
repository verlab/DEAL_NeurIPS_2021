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


import os, sys
import torch.optim as optim
import torch.nn.functional as F
import torch, kornia
from torchvision import transforms
import numpy as np
import cv2, math

import tqdm

from modules import losses
import matplotlib.pyplot as plt

from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter


logdir = None
save = None
writer = None


class DEAL:
	'''
		Class that defines the 'deal' descriptor abstraction similar to OpenCV interface
	'''
	def __init__(self, model_path, sift = True):
		self.model_path = model_path
		sys.path.append(os.path.dirname(model_path))
		import TPS_Transformer
		useResNet = False if 'ablation-ptn' in model_path else True
		self.net = TPS_Transformer.BaseModel(useResNet = useResNet, pretrainedResNet = False)
		self.net.load_state_dict(torch.load(model_path))
		self.sift = sift
		self.toTensor = transforms.ToTensor()
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.net = self.net.to(self.device)
		self.net.eval()

	def IC_angle(self, img, keypoints):
		for kp in keypoints:
			kp.angle = self.intensity_centroid(img,kp)

	def intensity_centroid(self, img, kp):
		''' Returns the angle of the keypoint [0,360] '''
		cx, cy = 0, 0
		pt = (int(kp.pt[0]), int(kp.pt[1]))

		halfPatchSize2 = 15 * 15 #same used in GEOBIT implementation
		for y in range(-15, 16):
			for x in range(-15, 16):
				d = x*x + y*y
				if d > 0 and  d < halfPatchSize2 and x + pt[0] >=0 and x + pt[0] < img.shape[1] and \
				y + pt[1] >=0 and y + pt[1] < img.shape[0]: # circular patch around the keypoint
					cx += (x/math.sqrt(d)) * img[y + pt[1], x + pt[0]]
					cy += (y/math.sqrt(d)) * img[y + pt[1], x + pt[0]]
				
		return cv2.fastAtan2(cy/255.0, cx/255.0)	

	def prepare_input(self, img, kps):
		dev = self.device
		X = [self.toTensor(img)]
		x_kps = [np.array([(k.pt[0], k.pt[1], k.angle, k.size) for k in kps], dtype = np.float32)]

		X_dict = {}

		nkps_x = []
		nori_x, nscale_x = [], []
		idx_keys = []
		Hs_x, Ws_x = [], []

		for b in range(1): #for each image in the batch, prepare keypoints -- TODO, for now only consider 1 image
			H, W = X[b].shape[-2:]
			imgSize = np.array([W-1,H-1], dtype = np.float32)
			nkp_x = x_kps[b][:,:2] / imgSize * 2 - 1
			Hs_x += [H] * len(x_kps[b])
			Ws_x += [W] * len(x_kps[b]) 

			idx = [b] * len(x_kps[b])
			idx_keys+=idx
			nkps_x.append(nkp_x)
			nori_x.append(x_kps[b][:,2])
			nscale_x.append(x_kps[b][:,3])

			X_dict[b] = {'img':X[b].to(dev).mean(0, keepdim = True)}
			
		nkps_x = np.vstack(nkps_x)
		scale_x = np.concatenate(nscale_x)
		#ori_x = np.concatenate(nori_x)
		ori_x = np.zeros_like(scale_x) 

		Hs_x, Ws_x = np.array(Hs_x, dtype = np.float32), np.array(Ws_x, dtype = np.float32)

		theta_x =   [torch.from_numpy(nkps_x).to(dev), 
					 torch.from_numpy(scale_x).to(dev),
					 torch.from_numpy(ori_x).to(dev),
					 torch.from_numpy(Hs_x).to(dev),
					 torch.from_numpy(Ws_x).to(dev)]

		return X_dict, theta_x, idx_keys	

	def compute(self, img, keypoints, return_warped = False):

		if isinstance(img, str):
			img = cv2.imread(img, 0)

		if not self.sift:
			self.IC_angle(img, keypoints) # Find keypoint angles

		X_dict, theta_x, idx_keys = self.prepare_input(img, keypoints)
		warped = self.net(X_dict, theta_x, idx_keys)

		descriptors = self.net.hardnet(warped)
		descriptors = descriptors.detach().cpu().numpy()

		if return_warped:
			return descriptors, warped.detach().cpu().permute(0,2,3,1).numpy()

		return descriptors



def grad_norm(parameters, norm_type = 2.0):
	from torch._six import inf
	if isinstance(parameters, torch.Tensor):
		parameters = [parameters]

	parameters = [p for p in parameters if p.grad is not None]
	norm_type = float(norm_type)
	if len(parameters) == 0:
		return torch.tensor(0.)
	device = parameters[0].grad.device
	if norm_type == inf:
		total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
	else:
		total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)

	return total_norm


def plot_grid(warped, title, mpl):
	#visualize 

	g = None
	n = warped[0].shape[0]

	for i in range(0, n, 16):
		if i + 16 <= n:
			for w in warped:
				pad_val = 0.7 if i//16%2 == 1 else 0
				gw = make_grid(w[i:i+16].detach().clone().cpu(), padding=4, pad_value=pad_val, nrow=16)
				g = gw if g is None else torch.cat((g, gw), 1)

	if mpl:
		plt.figure(figsize = (20,20))
		plt.imshow(g.permute(1,2,0).numpy()[...,::-1])
		plt.show()

	if logdir is not None and writer is not None:
		writer.add_image(title, g)




def train(net, dataset, nepochs=100, lr=1e-4, mpl = True, resume = None):
	global logdir, writer, save
	writer = SummaryWriter(logdir)

	opt = optim.Adam(filter(lambda x: x.requires_grad, net.parameters()) , lr=lr, weight_decay = 1e-4)
	scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.9)
	loss_vals = []

	if resume is not None:
		print('Resuming training from state...')
		net.load_state_dict(torch.load(resume + '.pth'))
		opt.load_state_dict(torch.load(resume + '-optim.pth'))
		scheduler.load_state_dict(torch.load(resume + '-scheduler.pth'))

	net.train()

	chunk_size = dataset.chunk_size if hasattr(dataset, 'chunk_size') else 1
	passes = dataset.passes if hasattr(dataset, 'passes') else 1
	epoch=0

	for subepoch in range(nepochs * chunk_size * passes):
		epoch = subepoch // (chunk_size * passes)
		net.train()
		train_loss = 0.
		ssim_val = 0.
		cnt = 0
		with tqdm.tqdm(total=len(dataset)) as pbar:
			for batch in dataset:
				if dataset.geo:
					X_dict, Y_dict, theta_x, theta_y , x_geo, y_geo, idx_keys = batch
				else:
					X_dict, Y_dict, theta_x, theta_y , idx_keys = batch

				opt.zero_grad()

				x_warped = net(X_dict, theta_x, idx_keys)
				y_warped = net(Y_dict, theta_y, idx_keys)

				# Photometric error loss
				# SSIM = losses.SSIM_loss(x_warped, y_warped)
				# sharpness_x =  losses.sharpness_loss(x_warped)
				# sharpness_y = losses.sharpness_loss(y_warped)
				# loss = (SSIM + sharpness_x + sharpness_y)/3.0

				# Photometric GEOPATCH error loss
				if dataset.geo:
					SSIM = (losses.SSIM_loss(x_warped, x_geo, 7) + losses.SSIM_loss(y_warped, y_geo, 7)) / 2.
				else:
					SSIM = losses.SSIM_loss(x_warped.detach(), y_warped.detach())

				#HardNet descriptor loss
				x_desc = net.hardnet(x_warped)
				y_desc = net.hardnet(y_warped)

				triplet_loss = losses.hardnet_loss(x_desc, y_desc) #hardest in-batch triplet margin ranking loss
				

				loss = SSIM + triplet_loss if dataset.geo else triplet_loss
				loss.backward()
				#writer.add_scalar('Gradient norm', grad_norm(net.parameters()), global_cnt)
				#torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
				opt.step()  

				train_loss += loss.detach().item()
				ssim_val += 1.0 - SSIM.detach().item()
				cnt+=1

				pbar.set_description('(Epoch {:d}/{:d} - #step {:d}) - Loss: {:.4f} - SSIM: {:.4f}'.format(epoch, nepochs, subepoch,
																												train_loss / cnt, ssim_val / cnt))
				pbar.update(1)
		# loss_vals.append({'loss': train_loss / cnt,
		#                   'ssim': ssim_val / cnt})
		writer.add_scalar('Training loss', train_loss/cnt, subepoch+1)
		writer.add_scalar('SSIM value', ssim_val/cnt, subepoch+1)
		if save is not None and subepoch%40 == 0:
			torch.save(net.state_dict(), os.path.join(save, '{:d}.pth'.format(subepoch)))
			torch.save(opt.state_dict(), os.path.join(save, '{:d}-optim.pth'.format(subepoch)))
			torch.save(scheduler.state_dict(), os.path.join(save, '{:d}-scheduler.pth'.format(subepoch)))
		scheduler.step()

		if not dataset.geo:
			plot_grid((x_warped[:16, ...], y_warped[:16, ...])
				, 'epoch {:d}/sample from subepoch {:d}'.format(epoch, subepoch), mpl)
		else:
			plot_grid((x_warped[:16, ...], y_warped[:16, ...], x_geo[:16, ...], y_geo[:16,...])
				, 'epoch {:d}/sample from subepoch {:d}'.format(epoch, subepoch), mpl)			


	# Plot rectified patches after training
	net.eval()

	figtitle = 'Samples after {:d} epochs'.format(nepochs)
	for i in range(0, len(dataset)):
		batch = dataset[i]
		if dataset.geo:
			X_dict, Y_dict, theta_x, theta_y , x_geo, y_geo, idx_keys = batch
		else:
			X_dict, Y_dict, theta_x, theta_y , idx_keys = batch

		x_warped = net(X_dict, theta_x, idx_keys)
		y_warped = net(Y_dict, theta_y, idx_keys)

		# plot some random pairs for visualization purposes
		rnd_idx = np.random.randint(0, x_warped.shape[0], 128)

		if not dataset.geo:
			plot_grid((x_warped[rnd_idx], y_warped[rnd_idx])
				, 'Samples after {:d} epochs'.format(epoch), mpl)
		else:
			plot_grid((x_warped[rnd_idx], y_warped[rnd_idx], x_geo[rnd_idx], y_geo[rnd_idx])
				, 'Samples after {:d} epochs'.format(epoch), mpl)

		break

def alt_train(net, dataset, nepochs=100, lr=1e-4, mpl = True, alt = 10):
	'''
	Train warper and hardnet in alternate mode, optimizing warper for some steps and then
	hardnet for some steps
	'''
	global logdir, writer, save
	writer = SummaryWriter(logdir)

	def freeze(net):
		for p in net.parameters(): p.requires_grad = False
	def unfreeze(net):
		for p in net.parameters(): p.requires_grad = True	

	freeze(net.hardnet)
	opt_warper = optim.Adam(filter(lambda x: x.requires_grad, net.parameters()) , lr=1e-4)
	scheduler_warper =  torch.optim.lr_scheduler.StepLR(opt_warper, step_size=30, gamma=0.9)
	unfreeze(net.hardnet)

	freeze(net.resnet) ; freeze(net.tps_transform)
	opt_hardnet = optim.SGD(filter(lambda x: x.requires_grad, net.parameters()) , lr=0.1, weight_decay = 1e-4)
	scheduler_hardnet =  torch.optim.lr_scheduler.StepLR(opt_hardnet, step_size=30, gamma=0.9)
	unfreeze(net.resnet) ; unfreeze(net.tps_transform)

	loss_vals = []

	chunk_size = dataset.chunk_size if hasattr(dataset, 'chunk_size') else 1
	passes = dataset.passes if hasattr(dataset, 'passes') else 1

	for subepoch in range(nepochs * chunk_size * passes):
		epoch = subepoch // (chunk_size * passes)
		net.train()
		train_loss = 0.
		ssim_val = 0.
		cnt = 0

		if subepoch % alt ==0 and (subepoch//alt) % 2 == 0:
			unfreeze(net.hardnet)
			freeze(net.resnet) ; freeze(net.tps_transform)
			opt = opt_hardnet ; scheduler = scheduler_hardnet
			print('training hardnet...')
		if subepoch % alt ==0 and (subepoch//alt) % 2 == 1:
			freeze(net.hardnet)
			unfreeze(net.resnet) ; unfreeze(net.tps_transform)
			opt = opt_warper; scheduler = scheduler_warper
			print('training tps...')

		with tqdm.tqdm(total=len(dataset)) as pbar:
			for batch in dataset:

				X_dict, Y_dict, theta_x, theta_y , idx_keys = batch

				opt.zero_grad()

				x_warped = net(X_dict, theta_x, idx_keys)
				y_warped = net(Y_dict, theta_y, idx_keys)

				# Photometric error loss
				# SSIM = losses.SSIM_loss(x_warped, y_warped)
				# sharpness_x =  losses.sharpness_loss(x_warped)
				# sharpness_y = losses.sharpness_loss(y_warped)
				# loss = (SSIM + sharpness_x + sharpness_y)/3.0

				#HardNet descriptor loss
				x_desc = net.hardnet(x_warped)
				y_desc = net.hardnet(y_warped)
				loss = losses.hardnet_loss(x_desc, y_desc) #hardest in-batch triplet margin ranking loss
				SSIM = losses.SSIM_loss(x_warped.detach(), y_warped.detach())


				loss.backward()
				#writer.add_scalar('Gradient norm', grad_norm(net.parameters()), global_cnt)
				#torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
				opt.step()  

				train_loss += loss.detach().item()
				ssim_val += 1.0 - SSIM.detach().item()
				cnt+=1

				pbar.set_description('(Epoch {:d}/{:d} - #step {:d}) - Loss: {:.4f} - SSIM: {:.4f}'.format(epoch, nepochs, subepoch,
																												train_loss / cnt, ssim_val / cnt))
				pbar.update(1)
		# loss_vals.append({'loss': train_loss / cnt,
		#                   'ssim': ssim_val / cnt})
		writer.add_scalar('Training loss', train_loss/cnt, subepoch+1)
		writer.add_scalar('SSIM value', ssim_val/cnt, subepoch+1)
		if save is not None and subepoch%40 == 0:
			torch.save(net.state_dict(), os.path.join(save, '{:d}.pth'.format(subepoch)))
		scheduler.step()
		plot_grid(x_warped[:16, ...], y_warped[:16, ...], 'epoch {:d}/sample from subepoch {:d}'.format(epoch, subepoch), mpl)


	# Plot rectified patches after training
	net.eval()

	figtitle = 'Samples after {:d} epochs'.format(nepochs)
	for i in range(0, len(dataset)):
		batch = dataset[i]
		X_dict, Y_dict, theta_x, theta_y , idx_keys = batch
		x_warped = net(X_dict, theta_x, idx_keys)
		y_warped = net(Y_dict, theta_y, idx_keys)
		# plot some random pairs for visualization purposes
		rnd_idx = np.random.randint(0, x_warped.shape[0], 128)
		plot_grid(x_warped[rnd_idx], y_warped[rnd_idx], title = figtitle, mpl = mpl)
		break
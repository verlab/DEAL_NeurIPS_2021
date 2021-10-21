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

import argparse
from datetime import datetime, timedelta
import os
from modules.models.HomographyDataset import AugmentationPipe, Phototour
import cv2, time
import torch
import numpy as np

from modules.models import TPS_Transformer as TPS_Module
from modules.models.TPS_Transformer import TPS_Transformer, BaseModel
from modules.NRDataset import NRDataset
from modules import utils

now = datetime.now() - timedelta(hours=3)
str_date = now.strftime("%d-%m-%Y--%HH_%MM_%SS")

def check_dir(f):
	if not os.path.exists(f):
		os.makedirs(f)

def parseArg():
	parser = argparse.ArgumentParser()
	parser.add_argument("-m", "--mode", help="Train or test"
	, required=True, choices = ['train', 'test'])
	parser.add_argument("-d", "--dataset", help="Which dataset to use"
	, required=False, choices = ['homography', 'nonrigid'], default = 'homography')
	parser.add_argument("-dpath", "--datapath", help="Dataset path."
	, required=False, default = './tests/img') 	
	parser.add_argument("-log", "--logdir", help="Output path where results will be saved."
	, required=False, default = './logdir') 
	parser.add_argument("-n", "--name", help="Run name"
	, required=False, default = str_date) 
	parser.add_argument("-e", "--epochs", help="Number of epochs", type=int
	, required=False, default = 10) 
	parser.add_argument("-s", "--save", help="Path for saving model"
	, required=False, default = './models') 	
	parser.add_argument("-r", "--resume", help="Path for resuming a training state"
	, required=False, default = None)
	parser.add_argument("--geo", help="Use geopatches in training"
	, action = 'store_true')
	parser.add_argument("--notps", help="Disable TPS transformer"
	, action = 'store_true')
	parser.add_argument("--pretrained", help="Use pretrained resnet model"
	, action = 'store_true')	

	args = parser.parse_args()
	if args.logdir is not None and not os.path.exists(args.logdir):
		raise RuntimeError(args.logdir + ' does not exist!')

	if args.geo and args.notps:
		raise RuntimeError('Invalid combination geo and notps')

	return args

def get_nb_trainable_params(model):
	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	nb_params = sum([np.prod(p.size()) for p in model_parameters])
	return nb_params

if __name__ == '__main__':
	args = parseArg()

	if args.mode == 'train':
		if args.dataset == 'homography':
			dataset = Phototour('./tests/img', n = 256, batch_size = 8)
		else:
			if os.path.exists(args.datapath) and args.datapath.endswith('.h5'):
				dataset = NRDataset(args.datapath, grayScale = True, batch_size = 8,
								   						 chunk_size = 32, passes=10, geo=args.geo)
			else:
				raise RuntimeError(args.datapath+' must exist and be an h5 file!')
		
		useResNet = False if args.notps else True
		net = BaseModel(useResNet = useResNet, pretrainedResNet = args.pretrained)

		print('Number of trainable parameters: {:d}'.format(get_nb_trainable_params(net)))
		check_dir(args.logdir)

		utils.logdir = os.path.join(args.logdir, args.name)
		utils.save = os.path.join(args.save, args.name)

		#Save architecture class
		check_dir(utils.save) 
		with open(TPS_Module.__file__, 'r') as f: arq_str = f.read()
		with open(os.path.join(utils.save, os.path.basename(TPS_Module.__file__)), 'w') as f:
			f.write(arq_str)
		print('Using log dir: ', utils.logdir)
		
		#utils.alt_train(net, dataset = dataset, nepochs = args.epochs, lr = 1e-4, mpl = False) #lr 1e-5
		utils.train(net, dataset = dataset, nepochs = args.epochs, lr = 0.5 * 1e-4, mpl = False, resume = args.resume)

		# for param_tensor in net.state_dict():
		# 	print(param_tensor, "\t", net.state_dict()[param_tensor].size())
		torch.save(net.state_dict(), os.path.join(args.save, args.name, args.name) + '.pth')

	elif args.mode == 'test':
		nrlfeat = utils.NRLFeat()
		dummy_img = np.ones((1080,1920), dtype = np.uint8)
		dummy_kps = [cv2.KeyPoint(500,500, 12.0, 90., 1.) for i in range(2048)]
		for i in range(30):
			descs = nrlfeat.compute(dummy_img, dummy_kps)
			print(str(i)+' ', end='', flush=True)
		print(descs.shape)
		print('done.')

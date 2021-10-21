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
import torch.nn.functional as F
import torch.nn as nn

class Pad2D(torch.nn.Module):
	def __init__(self, pad, mode): 
		super().__init__()
		self.pad = pad
		self.mode = mode

	def forward(self, x):
		return F.pad(x, pad = self.pad, mode = self.mode)


class HardNet(nn.Module):
	def __init__(self, nchannels=1):
		super().__init__()

		self.nchannels = nchannels

		self.features = nn.Sequential(
			nn.InstanceNorm2d(self.nchannels),
			Pad2D(pad=(0,0,1,1), mode = 'circular'),
			nn.Conv2d(self.nchannels, 32, 3, bias=False, padding=(0,1)),
			nn.BatchNorm2d(32, affine=False),
			nn.ReLU(),
			Pad2D(pad=(0,0,1,1), mode = 'circular'),
			nn.Conv2d(32, 32, 3, bias=False, padding=(0,1)),
			nn.BatchNorm2d(32, affine=False),
			nn.ReLU(),
			Pad2D(pad=(0,0,1,1), mode = 'circular'),
			nn.Conv2d(32, 64, 3, bias=False, padding=(0,1), stride=2),
			nn.BatchNorm2d(64, affine=False),
			nn.ReLU(),
			Pad2D(pad=(0,0,1,1), mode = 'circular'),
			nn.Conv2d(64, 64, 3, bias=False, padding=(0,1)),
			nn.BatchNorm2d(64, affine=False),
			nn.ReLU(),
			Pad2D(pad=(0,0,1,1), mode = 'circular'),
			nn.Conv2d(64, 128, 3, bias=False, padding=(0,1), stride=2),
			nn.BatchNorm2d(128, affine=False),
			nn.ReLU(),
			Pad2D(pad=(0,0,1,1), mode = 'circular'),
			nn.Conv2d(128, 128, 3, bias=False, padding=(0,1)),
			nn.BatchNorm2d(128, affine=False),
			nn.ReLU(),
			nn.Dropout(0.1),
			# nn.Conv2d(128, 128, 8, bias=False),
			# nn.BatchNorm2d(128, affine=False)
			#Rotation invariance block - pool angle axis
			nn.AvgPool2d((8,1), stride=1),
			nn.Conv2d(128, 128, (1,3), bias=False),
			nn.BatchNorm2d(128, affine=False),
			nn.Conv2d(128, 128, (1,3), bias=False),
			nn.BatchNorm2d(128, affine=False),
			nn.Conv2d(128, 128, (1,3), bias=False),
			nn.BatchNorm2d(128, affine=False),			
			nn.Conv2d(128, 128, (1,2), bias=False),
			nn.BatchNorm2d(128, affine=False)			
		)
		
	def forward(self, x):
		x = self.features(x).squeeze(-1).squeeze(-1)
		x = F.normalize(x)
		return x

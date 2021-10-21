# Copyright 2019 EPFL, Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file is a modified version from log-polar-descriptors, originally available at:
# https://github.com/cvlab-epfl/log-polar-descriptors/blob/master/modules/ptn/pytorch/models.py

# This modified version include three main improvements:
# 1. No requirement for images being at a predefined size or aspect ratio anymore
# 2. Rotation is directly implemented in the grid sampling generation, theres no need for tensor rolling
# 3. No requirement that the #nb of keypoints are perfect square to allow efficient F.grid_sample interpolation, we simply expand the image tensor in the batch dimension
#     by the #nb of keypoints without using additional memory


import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformer(nn.Module):
    def __init__(self,
                 transform="cartesian",
                 coords="linear",
                 resolution=32,
                 SIFTscale=1.0,
                 identity = False,
                 featureMap = False,
                 ):
        super(SpatialTransformer, self).__init__()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        #if torch.cuda.is_available() and onGPU: torch.cuda.set_device(0)

        self.coords = coords  # "linear" or "log"
        self.resolution = resolution  # e.g. 12.0
        self.SIFTscale = SIFTscale  # constant scale multiplier for all keypoints, e.g. 12 for opencv SIFT scale
        self.transform = transform
        self.identity = identity
        self.featureMap = featureMap

        self.rescalePTNtoSIFT = 0.5
        self.pi = torch.Tensor([np.pi]).to(self.device)
        self.normalize = None
        self.batchSize = None

        self.toRadC = np.pi / 180.

    # Cartesian Sampling Function
    def stn(self, img, theta, imgIDs=None):

      grid = self.generateCartGrid(theta)

      if self.dictProcess and imgIDs is not None:
        # allocate space for extracted patches, take care to preserve the patch ordering
        patches = torch.empty(self.getGridSize()).to(self.device)

        for key, val in img.items():
            val = val['feat' if self.featureMap else 'img'].unsqueeze(0)
            # get indices of all keypoints in batch placed on the current image
            indices = [
                idx for idx, idStr in enumerate(imgIDs) if idStr == key
            ]

            patches[indices, ...] = F.grid_sample(
                val.expand(len(indices),-1,-1,-1), grid[indices, ...], align_corners = True, padding_mode = 'border'
            )  # do bilinear interpolation to sample values on the grid

      else:
          patches = F.grid_sample(
              img, polargrid,
              align_corners = True
          )  # do bilinear interpolation to sample values on the grid 

      return patches


    #Polar Sampling Function
    def ptn(self, img, theta, imgIDs=None):
        #test = polargrid.permute(0,3,1,2)
        #test = F.interpolate(test, (16,16))
        #polargrid = F.interpolate(test, (32,32)).permute(0,2,3,1)
        #print('SHAPE:', test.shape)

        polargrid = self.generatePolarGrid(theta, self.coords)

        if self.dictProcess and imgIDs is not None:
            # allocate space for extracted patches, take care to preserve the patch ordering
            patches = torch.empty(self.getGridSize()).to(self.device)

            for key, val in img.items():
                val = val['img'].unsqueeze(0)
                # get indices of all keypoints in batch placed on the current image
                indices = [
                    idx for idx, idStr in enumerate(imgIDs) if idStr == key
                ]

                patches[indices, ...] = F.grid_sample(
                    val.expand(len(indices),-1,-1,-1), polargrid[indices, ...], align_corners = True, padding_mode = 'border'
                )  # do bilinear interpolation to sample values on the grid

        else:
            patches = F.grid_sample(
                img, polargrid,
                align_corners = False
            )  # do bilinear interpolation to sample values on the grid

        return self.normalize(patches)


    def getGridSize(self):
        gridSize = torch.Size((self.batchSize, self.nchannels, self.resolution, self.resolution))
        return gridSize

    def generateCartGrid(self, theta, nchannels = None):
      
      if self.identity:
        scaling = torch.ones_like(theta[1]) * (self.resolution)
        rotation = torch.zeros_like(theta[2])
      else:
        scaling = theta[1]
        rotation = theta[2]*self.toRadC

      if nchannels is not None:
        self.nchannels = nchannels

      self.batchSize = len(rotation)

      kpLoc = theta[0]
      Hs = theta[3]
      Ws = theta[4]

      rescaleSTNtoSIFT = 1.0 / Ws
      scaling = rescaleSTNtoSIFT * self.SIFTscale * scaling

      aspectRatio = Ws / Hs

      # get [batchSize x 2 x 3] affine transformation matrix
      affMat = torch.empty(len(scaling), 2, 3).to(self.device)

      affMat[:, 0, 0] = torch.cos(rotation) * scaling
      affMat[:, 0, 1] = -scaling * torch.sin(rotation)
      affMat[:, 0, 2] = kpLoc[:, 0]
      affMat[:, 1, 0] = scaling * aspectRatio * torch.sin(rotation)
      affMat[:, 1, 1] = torch.cos(rotation) * scaling * aspectRatio
      affMat[:, 1, 2] = kpLoc[:, 1]

      grid = F.affine_grid(
        affMat, self.getGridSize(), align_corners = True)  # get [-1 x 1]² grids and apply affine transformations

      return grid  


    def generatePolarGrid(self, theta, coords, nchannels = None):
      
      kpLoc = theta[0]
      scaling = theta[1]
      rotation = theta[2]*self.toRadC
      Hs = theta[3]
      Ws = theta[4]

      radius_factor = self.rescalePTNtoSIFT * self.SIFTscale * scaling

      maxR = radius_factor  # rescale W

      self.batchSize = len(rotation)
      
      if nchannels is not None:
        self.nchannels = nchannels

      # get grid resolution, assumes identical resolutions across dictionary if processing dictionary

      gridSize = self.getGridSize()  

      # get [self.batchSize x self.resolution x self.resolution x 2] grids with values in [-1,1],
      # define grids or call torch function and apply unit transform
      ident = torch.from_numpy(
          np.array(self.batchSize * [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype = np.float32)).to(self.device)
      grid = F.affine_grid(ident, gridSize, align_corners= False)
      grid_y = grid[:, :, :, 0].view(self.batchSize, -1)
      grid_x = grid[:, :, :, 1].view(self.batchSize, -1)

      maxR = torch.unsqueeze(maxR, -1).expand(-1, grid_y.shape[-1]).float().to(self.device)
      Hs = torch.unsqueeze(Hs, -1).expand(-1, grid_y.shape[-1])
      Ws = torch.unsqueeze(Ws, -1).expand(-1, grid_x.shape[-1])

      # get radius of polar grid with values in [1, maxR]
      normGrid = (grid_y + 1) / 2
      if coords == "log": r_s_ = torch.exp(normGrid * torch.log(maxR))
      elif coords == "linear": r_s_ = 1 + normGrid * (maxR - 1)
      else: raise RuntimeError('Invalid coords type, choose [log, linear]')

      # convert radius values to [0, 2maxR/W] range
      r_s = (r_s_ - 1) / (maxR - 1) * 2 * maxR / Ws  # r_s equals r^{x^t/W} in eq (9-10)

      rotation = torch.unsqueeze(rotation, -1).expand(-1, grid_x.shape[-1])
      # y is from -1 to 1; theta is from 0 to 2pi
      t_s = (
          grid_x + 1
      ) * self.pi - rotation #correct angle  # tmin_threshold_distance_s equals \frac{2 pi y^t}{H} in eq (9-10)

      # use + kpLoc to deal with origin, i.e. (kpLoc[:, 0],kpLoc[:, 1]) denotes the origin (x_0,y_0) in eq (9-10)
      xLoc = torch.unsqueeze(kpLoc[:, 0], -1).expand(-1, grid_x.shape[-1])
      yLoc = torch.unsqueeze(kpLoc[:, 1], -1).expand(-1, grid_y.shape[-1])

      aspectRatio = Ws/Hs

      x_s = r_s * torch.cos(
          t_s
      ) + xLoc  # see eq (9) : theta[:,0] shifts each batch entry by the kp's x-coords
      y_s = r_s * torch.sin(
          t_s
      ) * aspectRatio + yLoc  # see eq (10): theta[:,1] shifts each batch entry by the kp's y-coords

      # tensorflow grid is of shape [self.batchSize x 3 x self.resolution**2],
      # pytorch grid is of shape [self.batchSize x self.resolution x self.resolution x 2]
      # x_s and y_s are of shapes [1 x self.resolution**2]
      # bilinear interpolation in tensorflow takes _interpolate(input_dim, x_s_flat, y_s_flat, out_size)

      # reshape polar coordinates to square tensors and append to obtain [self.batchSize x self.resolution x self.resolution x 2] grid
      polargrid = torch.cat(
          (x_s.view(self.batchSize, self.resolution, self.resolution, 1),
            y_s.view(self.batchSize, self.resolution, self.resolution, 1)),
          -1)
      
      return polargrid   


    def forward(self, input):
        img, theta = input[0], input[1]
        imgIDs = None if len(input) < 3 else input[2]

        self.dictProcess = isinstance(
            img, dict
        )  # if received a dictionary then look up images for memory-efficient processing
        self.batchSize = theta[0].shape[
            0]  # get batch size, i.e. number of keypoints to process
        
        self.nchannels = img[0]['feat' if self.featureMap else 'img'].shape[0]

        if self.normalize is None:
          self.normalize = nn.InstanceNorm2d(self.nchannels)

        if self.transform == 'polar':
          x = self.ptn(img, theta, imgIDs)  # transform the input via PTN and return patches + meshgrid
        elif self.transform == 'cartesian':
          x = self.stn(img, theta, imgIDs)  # transform the input via PTN and return patches + meshgrid
        else:
          raise RuntimeError('Invalid transform type: Chooose [cartesian, polar]')

        return x
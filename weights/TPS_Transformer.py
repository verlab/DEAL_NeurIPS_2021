import torch
import torch.nn.functional as F
from torchvision import models
from modules.tps import pytorch as TPS
from modules.models.SpatialTransformer import SpatialTransformer
#from modules.models.HardNet import HardNet
import torch.nn as nn

class TPS_Transformer(torch.nn.Module):
    def __init__(self, ctrlshape, useResNet): 
        super().__init__()
        self.ctrlshape = ctrlshape
        self.nctrl = ctrlshape[0]*ctrlshape[1]
        self.nparam = (self.nctrl + 2)
        self.useResNet = useResNet
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.T = SpatialTransformer(transform = 'cartesian', identity= True, resolution=5, featureMap=True, 
                    SIFTscale = 1./8).to(self.device)

        self.loc = torch.nn.Sequential(
            #torch.nn.MaxPool2d(3),
            torch.nn.Flatten(),
            torch.nn.Dropout(0.15),
            torch.nn.Linear(5*5*128, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, self.nparam*2),
            torch.nn.Tanh()
        ).to(self.device)


        self.loc[-2].weight.data.zero_() #normal_(0, 1e-5)
        self.loc[-2].bias.data.zero_()
        
    def forward(self, polargrid, theta, imgs, imgIDs):
        N, H, W, _ = polargrid.shape
        C = imgs[0]['img'].shape[0]

        if self.useResNet:
          xt = self.T([imgs, theta, imgIDs])

          kfactor = 0.3
          offset = (1.0 - kfactor)/2.
          vmin = polargrid.view(N,-1,2).min(1)[0].unsqueeze(1).unsqueeze(1)
          vmax = polargrid.view(N,-1,2).max(1)[0].unsqueeze(1).unsqueeze(1)
          ptp = vmax - vmin
          polargrid = (polargrid - vmin) / ptp
          #scale by a factor and normalize to center for better condition
          polargrid = polargrid * kfactor + offset

          grid_img = polargrid.permute(0,3,1,2) # Trick to allow interpolating a torch 2D meshgrid into desired shape
          ctrl = F.interpolate(grid_img, self.ctrlshape).permute(0,2,3,1).view(N,-1,2)

          theta = self.loc(xt).view(-1, self.nparam, 2)

          I_polargrid = theta.new(N, H, W, 3) #create a new tensor with identity polar grid (normalized by keypoint attributes)
          I_polargrid[..., 0] = 1.0
          I_polargrid[..., 1:] = polargrid

          z = TPS.tps(theta, ctrl, I_polargrid)
          tps_warper = (I_polargrid[...,1:] + z) # *2-1
   
          #reverse transform - scale by a factor and normalize to center for better condition
          tps_warper = (tps_warper - offset) / kfactor        
          #denormalize each element in batch
          tps_warper = tps_warper * ptp + vmin

        ##### ablation without TPS #####
        #simply return identity polar grid transform
        else:
          tps_warper = polargrid
        ################################

        patches = torch.zeros((N,C,H,W), dtype = torch.float32).to(self.device)

        for key, val in imgs.items():
            val = val['img'].unsqueeze(0)
            # get indices of all keypoints in batch placed on the current image
            indices = [ idx for idx, idStr in enumerate(imgIDs) if idStr == key ]

            patches[indices, ...] = F.grid_sample(val.expand(len(indices),-1,-1,-1),
                                                  tps_warper[indices, ...], align_corners = False,  padding_mode = 'border') 
            

        return patches


class BaseModel(torch.nn.Module):
    def __init__(self, useResNet = False, pretrainedResNet = True, nchannels = 1):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nchannels = nchannels

        self.stn = SpatialTransformer(resolution=32, SIFTscale = 48.0).to(self.device) # 24.0
        self.tps_transform = TPS_Transformer(ctrlshape=(8,4), useResNet=useResNet).to(self.device)
        self.hardnet = HardNet(nchannels = self.nchannels).to(self.device)

        self.useResNet = useResNet

        if self.useResNet:
         resnet = models.resnet34(pretrained=pretrainedResNet, progress = False)
         self.resnet = nn.Sequential(*(list(resnet.children())[:6])).to(self.device) # get first N ConvBlocks from resnet

        
    def forward(self, imgs, theta, imgsIDs):

        if self.useResNet:
          for key, val in imgs.items():
            img = val['img'].unsqueeze(0)
            img = img.expand(-1,3,-1,-1) if self.nchannels == 1 else img
            imgs[key]['feat'] = self.resnet( img ).squeeze(0)

        meshgrid = self.stn.generatePolarGrid(theta, coords='linear', nchannels=imgs[0]['img'].shape[0])
        #meshgrid = self.stn.generateCartGrid(theta, nchannels=imgs[0]['img'].shape[0])

        out = self.tps_transform(meshgrid, theta, imgs, imgsIDs)
        return out



############################################ HardNet definition ####################################################

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

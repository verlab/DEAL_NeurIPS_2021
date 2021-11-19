dependencies = ['torch', 'os', 'requests', 'tensorboard', 'kornia']
import os, requests
from modules.utils import DEAL as DEAL_Model

def DEAL(sift=True, **kwargs):
    """ # This docstring shows up in hub.help()
    Resnet18 model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    model_folder = './models'

    pth_path = os.path.join(model_folder, 'newdata-DEAL-big.pth')
    net_path = os.path.join(model_folder, 'TPS_Transformer.py')
    
    if not(os.path.isfile(pth_path)):
        pth_link = 'https://github.com/verlab/DEAL_NeurIPS_2021/raw/main/models/newdata-DEAL-big.pth'
        r = requests.get(pth_link, allow_redirects=True)
        open(pth_path, 'wb').write(r.content)

    if not(os.path.isfile(net_path)):
        net_link = 'https://raw.githubusercontent.com/verlab/DEAL_NeurIPS_2021/main/models/TPS_Transformer.py'
        r = requests.get(net_link, allow_redirects=True)
        open(net_path, 'wb').write(r.content)


    model = DEAL_Model(pth_path, sift = sift) # Create the descriptor and load arctecture
    return model
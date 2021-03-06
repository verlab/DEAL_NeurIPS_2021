dependencies = ['torch', 'os', 'requests', 'cv2', 'numpy', 'tqdm', 'kornia']

def DEAL(sift=True, weights_folder = './models', **kwargs):
    """
    DEAL - Extracting Deformation-Aware Local Features by Learning to Deform 
    sift (bool): Indicate if SIFT keypoints are beeing used. If false, the model estimates keypoint rotation. Deafult = True
    weights_folder (str): Where to download and save the model's weights. Default = './models'

    to use:
    model.compute(image, keypoints) # just like opencv interface
    """
    import os, requests
    from modules.utils import DEAL as DEAL_Model

    pth_path = os.path.join(weights_folder, 'newdata-DEAL-big.pth')
    net_path = os.path.join(weights_folder, 'TPS_Transformer.py')
    
    if not os.path.isdir(weights_folder):
        os.makedirs(weights_folder)

    if not(os.path.isfile(pth_path)):
        print('Downloading weights...')
        pth_link = 'https://github.com/verlab/DEAL_NeurIPS_2021/raw/main/models/newdata-DEAL-big.pth'
        r = requests.get(pth_link, allow_redirects=True)
        open(pth_path, 'wb').write(r.content)

    if not(os.path.isfile(net_path)):
        print('Downloading TPS_Transformer...')
        net_link = 'https://raw.githubusercontent.com/verlab/DEAL_NeurIPS_2021/main/models/TPS_Transformer.py'
        r = requests.get(net_link, allow_redirects=True)
        open(net_path, 'wb').write(r.content)


    model = DEAL_Model(pth_path, sift = sift) # Create the descriptor and load arctecture
    return model
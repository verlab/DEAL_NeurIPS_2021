import cv2 # opencv to read images and extract keypoints
import torch

if __name__ == "__main__":
    deal_help = torch.hub.help('verlab/DEAL_NeurIPS_2021', 'DEAL', force_reload=True)
    print(deal_help)

    deal = torch.hub.load('verlab/DEAL_NeurIPS_2021', 'DEAL', True, './hub_model')
    sift = cv2.SIFT_create(2048)

    img_path = "./example/notredame.png"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    kps = sift.detect(img)
    desc = deal.compute(img, kps)

    print("Desc Shape:{}".format(desc.shape))
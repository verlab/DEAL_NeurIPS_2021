import cv2 # opencv to read images and extract keypoints
from modules.utils import DEAL #include the modules on your path


if __name__ == "__main__":
    net_path = 'weights/newdata-DEAL-big.pth' # weight path
    nrlfeat = DEAL(net_path, sift = True) # Create the descriptor and load arctecture
    sift = cv2.SIFT_create()

    img_path = "./example/notredame.png"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    kps = sift.detect(img)
    desc = nrlfeat.compute(img, kps)

    print("Desc Shape:{}".format(desc.shape))
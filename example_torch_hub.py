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
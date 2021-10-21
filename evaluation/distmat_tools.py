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


import numpy as np
import cv2 

def distMat(a,b):
    m = a.shape[0]
    n = b.shape[0]
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)
    A = np.repeat(a_norm[:, np.newaxis], n, axis = 1)
    B = np.repeat(b_norm[np.newaxis, :], m, axis = 0)
    x = a @ (b / b_norm[:, np.newaxis]).T
    D = np.sqrt((B-x)**2 + A**2 - x**2)
    return D

def save(desc_ref, desc_tgt, filename):
	desc_ref = np.array(desc_ref)
	desc_tgt = np.array(desc_tgt)

	dist_mat = distMat(desc_ref, desc_tgt)

	with open(filename, 'w') as f:
		f.write('%d %d\n'%(dist_mat.shape[0], dist_mat.shape[1]))
		for i in range(dist_mat.shape[0]):
			for j in range(dist_mat.shape[1]):
				f.write('%.5f '%(dist_mat[i,j]))
			
		f.write('\n')

def save_cvnorm(desc_ref, desc_tgt, filename):
	desc_ref = np.array(desc_ref)
	desc_tgt = np.array(desc_tgt)

	dist_mat = distMat(desc_ref, desc_tgt)

	for i in range(dist_mat.shape[0]):
		for j in range(dist_mat.shape[1]):
			dist_mat[i,j] = cv2.norm(desc_ref[i] - desc_tgt[j])

	with open(filename, 'w') as f:
		f.write('%d %d\n'%(dist_mat.shape[0], dist_mat.shape[1]))
		for i in range(dist_mat.shape[0]):
			for j in range(dist_mat.shape[1]):
				f.write('%.5f '%(dist_mat[i,j]))
			
		f.write('\n')	

def load_cv_kps(csv):
	keypoints = []
	for line in csv:
		k = cv2.KeyPoint(line['x'], line['y'], line['size']*1., line['angle'])
		keypoints.append(k)

	return keypoints
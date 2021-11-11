#!/usr/bin/env python
# coding: utf-8

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

import cv2

import os
import glob
import argparse
import numpy as np
import time

import tqdm
import distmat_tools

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

experiment_name = ''
exp_dir_target = ''


def check_dir(f):
	if not os.path.exists(f):
		os.makedirs(f)



def parseArg():
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input", help="Input path containing single or several (use -d flag) PNG-CSV dataset folders"
	, required=True, default = 'lists.txt') 
	parser.add_argument("-o", "--output", help="Output path where results will be saved."
	, required=True, default = '.') 
	parser.add_argument("-f", "--file", help="Use file list with several input dirs instead (make sure -i points to .txt path)"
	, action = 'store_true') 
	parser.add_argument("-d", "--dir", help="is a dir with several dataset folders?"
	, action = 'store_true')
	parser.add_argument("--sift", help="Run with SIFT keypoints instead of ground-truth .csv"
	, action = 'store_true')
	parser.add_argument("--tps_path", help="Directory with refined TPS path and keypoints"
	, required=False, default = '')
	parser.add_argument("--model", help="Path to trained model"
	, required=False, default = 'models/newdata-DEAL-big.pth')
	parser.add_argument("--exp-name", help="Name of the experiment"
	, required=False, default = 'DEAL')

	args = parser.parse_args()

	if args.sift and (args.tps_path == '' or not os.path.exists(args.tps_path)):
		raise RuntimeError('--sift requires a valid --tps_path folder')

	return args


def correct_old_csv(csv):
	for line in csv:
		if line['x'] < 0 or line['y'] < 0:
			line['valid'] = 0


def gen_keypoints_from_csv(csv):
	keypoints = []
	for line in csv:
		if line['valid'] == 1:
			k = cv2.KeyPoint(float(line['x']), float(line['y']),15.0, 0.0) 
			k.class_id = int(line['id'])
			keypoints.append(k)

	return keypoints	 

			
def get_dir_list(filename):
	with open(filename,'r') as f:
		dirs = [line.rstrip('\n').rstrip() for line in f if line.rstrip('\n').rstrip()]
	return dirs or False

def save_dist_matrix(ref_kps, ref_descriptors, ref_gt, tgt_kps, descriptors, tgt_gt, out_fname):
	#np.linalg.norm(a-b)
	print ('saving matrix in:',  out_fname)
	size = len(ref_gt)
	dist_mat = np.full((size,size),-1.0,dtype = np.float32)
	valid_m = 0
	matches=0

	matching_sum = 0

	begin = time.time()

	for m in range(len(ref_kps)):
		i = ref_kps[m].class_id
		if ref_gt[i]['valid'] and tgt_gt[i]['valid']:
			valid_m+=1
		for n in range(len(tgt_kps)):
			j = tgt_kps[n].class_id
			if ref_gt[i]['valid'] and tgt_gt[i]['valid'] and tgt_gt[j]['valid']:
				dist_mat[i,j] = np.linalg.norm(ref_descriptors[m]-descriptors[n]) #distance.euclidean(ref_d,tgt_d) #np.linalg.norm(ref_d-tgt_d)

	print('Time to match DEAL: %.3f'%(time.time() - begin))

	mins = np.argmin(np.where(dist_mat >= 0, dist_mat, 65000), axis=1)
	for i,j in enumerate(mins):
		if i==j and ref_gt[i]['valid'] and tgt_gt[i]['valid']:
			matches+=1

	print ('--- MATCHES --- %d/%d'%(matches,valid_m))

	with open(out_fname, 'w') as f:

		f.write('%d %d\n'%(size,size))

		for i in range(dist_mat.shape[0]):
			for j in range(dist_mat.shape[1]):
				f.write('%.8f '%(dist_mat[i,j]))


def run_benchmark(args):
	import sys, os
	deal_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
	sys.path.append(deal_dir)
	from modules.utils import DEAL
	net_path = args.model
	extractor = DEAL(net_path, sift = args.sift)
	exp_name = args.exp_name

	print('Running benchmark')

	ref_descriptor = None
	ref_gt = None

	if args.file:
		exp_list = get_dir_list(args.input)
	elif args.dir:
		exp_list = [d for d in glob.glob(args.input+'/*/*') if os.path.isdir(d)]
	else:
		exp_list = [args.input]

	exp_list = list(filter(lambda x: 'DeSurTSampled' in x or  'Kinect1' in x or 'Kinect2Sampled' in x or 'SimulationICCV' in x, exp_list))
	print("Found {} folders".format(len(exp_list)))
	for exp_dir in tqdm.tqdm(exp_list):

		dataset_name = os.path.join(*os.path.abspath(exp_dir).split('/')[-2:])

		experiment_files = glob.glob(exp_dir + "/*-rgb*")
		# print('Dataset: {}. Found {} exp files'.format(dataset_name, len(experiment_files)))
	
		master_f = ''
		for exp_file in experiment_files:
			if 'master' in exp_file or 'ref' in exp_file:
				fname = exp_file.split('-rgb')[0]

				if not args.sift:
					ref_gt = np.recfromcsv(fname + '.csv', delimiter=',', filling_values=np.nan, case_sensitive=True, deletechars='', replace_space=' ')
					correct_old_csv(ref_gt)
					ref_kps = gen_keypoints_from_csv(ref_gt)

				else:
					tps_fname = os.path.join(args.tps_path, *fname.split('/')[-3:])
					ref_gt = np.recfromcsv(tps_fname + '.sift', delimiter=',', filling_values=np.nan, case_sensitive=True, deletechars='', replace_space=' ')
					ref_kps = distmat_tools.load_cv_kps(ref_gt)
					
				img = fname + '-rgb.png'
				ref_descriptors = extractor.compute(img, ref_kps)
				master_f = exp_file

		for exp_file in experiment_files:

			if 'master' not in exp_file and 'ref' not in exp_file:
				fname = exp_file.split('-rgb')[0]
				if not args.sift:
					tgt_gt = np.recfromcsv(fname + '.csv', delimiter=',', filling_values=np.nan, case_sensitive=True, deletechars='', replace_space=' ')
					correct_old_csv(tgt_gt)
					tgt_kps = gen_keypoints_from_csv(tgt_gt)
				else:
					tps_fname = os.path.join(args.tps_path, *fname.split('/')[-3:])
					tgt_gt = np.recfromcsv(tps_fname + '.sift', delimiter=',', filling_values=np.nan, case_sensitive=True, deletechars='', replace_space=' ')
					tgt_kps = distmat_tools.load_cv_kps(tgt_gt)

				img = fname + '-rgb.png'
				descriptors = extractor.compute(img, tgt_kps)

				if not args.sift:
					mat_fname = os.path.basename(master_f).split('-rgb')[0] + '__' + os.path.basename(exp_file).split('-rgb')[0] + \
								'__' + exp_name + '.txt'
				else:
					mat_fname = os.path.basename(master_f).split('-rgb')[0] + '__' + os.path.basename(exp_file).split('-rgb')[0] + \
								'__' + exp_name	+ '.dist'			

				result_dir = os.path.join(args.output,experiment_name) + '/' + dataset_name + '/' + exp_dir_target
				check_dir(result_dir)

				if not args.sift:
					save_dist_matrix(ref_kps,ref_descriptors,ref_gt, tgt_kps, descriptors,tgt_gt, os.path.join(result_dir,mat_fname))
				else:
					distmat_tools.save(ref_descriptors, descriptors, os.path.join(result_dir,mat_fname))


if __name__ == "__main__":
	args = parseArg()
	run_benchmark(args)

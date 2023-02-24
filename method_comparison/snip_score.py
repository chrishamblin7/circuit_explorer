#given a feature this returns scores, no mask
import torch
import numpy as np
from PIL import Image
from collections import OrderedDict
import numpy as np
import time
import os
from circuit_explorer.utils import load_config
from circuit_explorer.data_loading import rank_image_data

##DATA LOADER###
import torch.utils.data as data
import torchvision.datasets as datasets
from circuit_explorer.data_loading import rank_image_data
from copy import deepcopy
from circuit_explorer.score import snip_score, structure_scores



import argparse


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--layer", type = str, 
						help='use layers = OrderedDict([*model.named_modules()])')
	parser.add_argument('--unit', type=int,help='numeral for unit in layer of target feature')
	parser.add_argument('--out-root', type=str,default ='./circuit_ranks/',help='root path to output folder')
	parser.add_argument("--config", type = str,default = '../configs/alexnet_sparse_config.py',help='relative_path to config file')
	parser.add_argument("--data-path", type = str, default = None, help='path to image data folder')
	parser.add_argument('--device', type = str, default='cuda:0', help='default "cuda:0"')  
	parser.add_argument('--batch-size', type=int, default=64,
						help='number of batches for dataloader')

	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = get_args()
	print(args)

	#get variables from config
	config = load_config(args.config)


	model = config.model


	layer = args.layer
	unit = args.unit
	device= args.device



	if args.data_path is None:
		data_path = config.data_path
	else:
		data_path = args.data_path

	if args.batch_size is None:
		batch_size = config.batch_size
	else:
		batch_size = args.batch_size


	imageset = data_path.split('/')[-1]
	if data_path[-1] == '/':
		imageset = data_path.split('/')[-2]



	kwargs = {'num_workers': config.num_workers, 'pin_memory': True, 'sampler':None} if 'cuda' in device else {}


	label_file_path =  config.label_file_path


	dataloader = data.DataLoader(rank_image_data(data_path,
												config.preprocess,
												label_file_path = label_file_path,class_folders=True),
												batch_size=batch_size,
												shuffle=False,
												**kwargs)

	
	start = time.time()


	#GET scores

	layers = OrderedDict([*model.named_modules()])
	model.to(device)
	scores = {}
	scores = snip_score(model,dataloader,layer,unit)
	scores = structure_scores(scores, model, structure='kernels')


	save_object = {'scores':scores,
				'layer':layer,
				'unit':unit,
				'structure':'kernels',
				'batch_size':batch_size,
				'data_path':data_path,
				'config':args.config
					}

	if not os.path.exists(args.out_root+'/'+config.name+'/'+imageset+'/snip'):
		os.makedirs(args.out_root+'/'+config.name+'/'+imageset+'/snip',exist_ok=True)
	torch.save(save_object,'%s/%s/%s/snip/%s_%s:%s.pt'%(args.out_root,config.name,imageset,config.name,layer,str(unit)))


	print(time.time() - start)

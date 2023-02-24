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
from circuit_explorer.score import force_score, structure_scores
from circuit_explorer.target import feature_target_saver



import argparse


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--layer", type = str, 
						help='use layers = OrderedDict([*model.named_modules()])')
	parser.add_argument('--unit', type=int,help='numeral for unit in layer of target feature')
	parser.add_argument('--T', type = int, default=10, help='number of FORCE iterations,default 10')
	parser.add_argument('--structure', type = str, default='kernels', help='default kernels')
	parser.add_argument('--sparsity', type=float, default=.5,
						help='number of batches for dataloader')
	parser.add_argument('--out-root', type=str,default ='/mnt/data/chris/nodropbox/Projects/circuit_explorer/correlations/',help='root path to output folder')
	parser.add_argument("--config", type = str,default = '../configs/alexnet_sparse_config.py',help='relative_path to config file')
	parser.add_argument("--data-path", type = str, default = None, help='path to image data folder')
	parser.add_argument('--device', type = str, default='cuda:0', help='default "cuda:0"')  
	parser.add_argument('--batch-size', type=int, default=64,
						help='number of batches for dataloader')
	parser.add_argument("--original_act_file", type = str, default = None, help='path to original activations file')



	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = get_args()
	print(args)

	#get variables from config
	config = load_config(args.config)


	model = config.model


	target_layer = args.layer
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


	#original_activations
	if args.original_act_file is None:

		original_activations = []
		print('getting original activations')
		#we save target activations in a context that allows us to handle the annoying problem of dangling hooks
		with feature_target_saver(model,target_layer,unit) as target_saver:
			#then we just run our data through the model, the target_saver will store activations for us
			for i, data in enumerate(dataloader, 0):
				inputs, labels = data
				inputs = inputs.to(device)
				target_activations = target_saver(inputs)
				#the target_saver doesnt aggregate activations, it overwrites each batch, so we need to save our data
				original_activations.append(target_activations.detach().cpu().type(torch.FloatTensor))

			#turn batch-wise list into concatenated tensor
			original_activations = torch.cat(original_activations)
	else:
		print('fetching original activations from file: %s'%args.original_act_file)
		original_activations = torch.load(args.original_act_file)
		original_activations = original_activations[target_layer+':'+str(unit)]		


	#this also applies the mask
	scores = force_score(model,dataloader,target_layer,unit,
						 keep_ratio= args.sparsity,T=args.T,
						 apply_final_mask=True)

	circuit_activations = []
	with feature_target_saver(model,target_layer,unit) as target_saver:

		#then we just run our data through the model, the target_saver will store activations for us
		for i, data in enumerate(dataloader, 0):
			model.zero_grad()

			inputs, target = data
			inputs = inputs.to(device)
			target = target.to(device)

			target_activations = target_saver(inputs)
			#import pdb; pdb.set_trace()
			#loss = sum_abs_loss(target_activations)
			#loss.backward()
			
			#save activations
			circuit_activations.append(target_activations.detach().cpu().type(torch.FloatTensor))
			

		#turn batch-wise list into concatenated tensor
		circuit_activations = torch.cat(circuit_activations)



	#correlations
	correlation = np.corrcoef(circuit_activations.flatten(),original_activations.flatten())[0][1]


	#saving
	out_folder = '%s/%s/%s/force/'%(args.out_root,config.name,imageset)
	out_file = '%s_%s:%s.pt'%(config.name,target_layer,str(unit))
	out_path = out_folder+out_file

	if os.path.exists(out_path):
		save_object = torch.load(out_path)
		save_object['correlations'].append(correlation)
		save_object['sparsities'].append(args.sparsity)
	else:
		save_object = {'correlations':[correlation],
					   'sparsities':[args.sparsity],
					   'unit':unit,
				       'layer':target_layer
					  }

	if not os.path.exists(out_folder):
		os.makedirs(out_folder,exist_ok=True)
	torch.save(save_object,out_path)

	print(time.time() - start)
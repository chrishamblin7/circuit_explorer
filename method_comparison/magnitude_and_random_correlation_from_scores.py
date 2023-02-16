import torch
import numpy as np
from PIL import Image
from collections import OrderedDict
import numpy as np
import os
from circuit_pruner.utils import load_config
from circuit_pruner.data_loading import rank_image_data
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from circuit_pruner.simple_api.mask import mask_from_scores, apply_mask,setup_net_for_mask
from circuit_pruner.simple_api.target import feature_target_saver, sum_abs_loss
from circuit_pruner.simple_api.score import magnitude_scores_from_scores, random_scores_from_scores
from time import time
import pickle


#params
name = 'inception'
dataset_name = 'imagenet_2'
config_file = '../configs/%s_config.py'%name
snip_scores_folder = '/mnt/data/chris/nodropbox/Projects/circuit_pruner/circuit_scores/%s/%s/snip/'%(name,dataset_name)
out_folder_root = '/mnt/data/chris/nodropbox/Projects/circuit_pruner/correlations/'
device = 'cuda:2'
batch_size = 40
sparsities = [.9,.8,.7,.6,.5,.4,.3,.2,.1,.05,.01,.005,.001]
original_activations_file = '/mnt/data/chris/nodropbox/Projects/circuit_pruner/original_activations/%s/%s/original_activations.pt'%(name,dataset_name)
save_activations = False
structure = 'kernels'
score_kinds = ['magnitude'] #choices are magnitude and random

#model
config = load_config(config_file)
model = config.model
_ = model.to(device).eval()


#dataloader
kwargs = {'num_workers': 4, 'pin_memory': True, 'sampler':None} if 'cuda' in device else {}
dataloader = torch.utils.data.DataLoader(rank_image_data(config.data_path,
										config.preprocess,
										label_file_path = config.label_file_path,class_folders=True),
										batch_size=batch_size,
										shuffle=False,
										**kwargs)


if not os.path.exists(out_folder_root):
	os.makedirs(out_folder_root, exist_ok=True)


#get data
score_files = os.listdir(snip_scores_folder)
for score_file in score_files: 
	start = time()

	print(score_file)
	#if os.path.exists(out_folder+score_file):
	#	print('skipping')
	#	continue
	score_dict = torch.load(os.path.join(snip_scores_folder,score_file))
	unit = score_dict['unit']
	target_layer = score_dict['layer']
	snip_scores = score_dict['scores']

	setup_net_for_mask(model) #reset mask in net

	#original_activations
	if original_activations_file is None:

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
		print('fetching original activations from file: %s'%original_activations_file)
		original_activations = torch.load(original_activations_file)
		original_activations = original_activations[target_layer+':'+str(unit)]	


	#get magnitude/random scores
	scores = {}
	for score_kind in score_kinds: 
		if score_kind == 'magnitude':
			scores[score_kind] = magnitude_scores_from_scores(snip_scores,model,target_layer,unit)
		elif score_kind == 'random':
			scores[score_kind] = random_scores_from_scores(snip_scores,target_layer,unit)
	del snip_scores

	for score_kind in score_kinds:
		folder_path = out_folder_root+'/'+name+'/'+dataset_name+'/'+score_kind+'/'
		if os.path.exists(folder_path+score_file):
			print('skipping')
			continue
		#get activations 
		print('getting %s score correlations'%score_kind)
		circuit_activations = []
		with feature_target_saver(model,target_layer,unit) as target_saver:
			for sparsity in sparsities:
			#for sparsity in sparsities[cat]:
				print(sparsity)

				#MASK THE MODEL
				mask = mask_from_scores(scores[score_kind],sparsity = sparsity,model=model,unit=unit,target_layer_name=target_layer)
				apply_mask(model,mask)

				activations = []
				#then we just run our data through the model, the target_saver will store activations for us
				for i, data in enumerate(dataloader, 0):
					model.zero_grad()

					inputs, target = data
					inputs = inputs.to(device)
					target = target.to(device)

					target_activations = target_saver(inputs)

					
					#save activations
					activations.append(target_activations.detach().cpu().type(torch.FloatTensor))
					

				#turn batch-wise list into concatenated tensor
				activations = torch.cat(activations)
				circuit_activations.append(activations)


		#correlations
		correlations = []
		for a in circuit_activations: 
			correlations.append(np.corrcoef(a.flatten(),original_activations.flatten())[0][1])

		#all_correlations[score_file] = correlations

		save_object = {'correlations':correlations,
					'sparsities':sparsities,
					'unit':unit,
					'layer':target_layer,
					'model':name,
					'method':score_kind,
					'model':name,
					'structure':structure}

		if save_activations:
			save_object['circuit_activations'] = circuit_activations
			save_object['original_activations'] = original_activations



		folder_path = out_folder_root+'/'+name+'/'+dataset_name+'/'+score_kind+'/'
		if not os.path.exists(folder_path):
			os.makedirs(folder_path,exist_ok=True)
		torch.save(save_object,folder_path+score_file)

		print(time()-start)




#torch.save(all_correlations,'resnet18_all_correlations.pt')
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from typing import Dict, Iterable, Callable
from collections import OrderedDict
from circuit_explorer.utils import TargetReached, params_2_target_from_scores
import types
from torch import nn, Tensor
from typing import Dict, Iterable, Callable
from copy import deepcopy
from math import log, exp, ceil
from circuit_explorer.target import feature_target_saver,sum_abs_loss, positional_loss
from circuit_explorer.mask import mask_from_scores, setup_net_for_mask, apply_mask
from circuit_explorer.dissected_Conv2d import *

####  SCORES ####

'''
Functions for computing saliency scores for parameters on models
'''

def snip_score(model,dataloader,target_layer_name,unit,layer_types_2_score = [nn.Conv2d,nn.Linear],loss_f = sum_abs_loss,absolute=True,use_weight_mask=False):

	_ = model.eval()
	device = next(model.parameters()).device  
	layers = OrderedDict([*model.named_modules()])

	#target_saver = feature_target_saver(model,layer_name,unit)
	scores = OrderedDict()
	with feature_target_saver(model,target_layer_name,unit) as target_saver:
		for i, data in enumerate(dataloader, 0):

			inputs, label = data
			inputs = inputs.to(device)
			#label = label.to(device)

			model.zero_grad() #very import!
			target_activations = target_saver(inputs)

			#feature collapse
			loss = loss_f(target_activations)
			loss.backward()

			#get weight-wise scores
			for layer_name,layer in layers.items():
				if type(layer) not in layer_types_2_score:
					continue
					
				if layer.weight.grad is None:
					continue

				if use_weight_mask: #does the model have a weight mask?
					#scale scores by batch size (*inputs.shape)
					if absolute:
						layer_scores = torch.abs(layer.weight_mask.grad).detach().cpu()*inputs.shape[0]
					else:
						layer_scores = (layer.weight_mask.grad).detach().cpu()*inputs.shape[0]

				else:
					if absolute:
						layer_scores = torch.abs(layer.weight*layer.weight.grad).detach().cpu()*inputs.shape[0]
					else:
						layer_scores = (layer.weight*layer.weight.grad).detach().cpu()*inputs.shape[0]


				
				if layer_name not in scores.keys():
					scores[layer_name] = layer_scores
				else:
					scores[layer_name] += layer_scores
					
	# # eliminate layers with stored but all zero scores
	remove_keys = []
	for layer in scores:
		if torch.sum(scores[layer]) == 0.:
			remove_keys.append(layer)
	if len(remove_keys) > 0: 
		print('removing layers from scores with scores all 0:')
		for k in remove_keys:
			print(k)
			del scores[k]
		  
	#target_saver.hook.remove() # this is important or we will accumulate hooks in our model
	model.zero_grad() 

	return scores


def force_score(model, dataloader,target_layer_name,unit,keep_ratio=.1, T=10, num_params_to_keep=None, structure='kernels',layer_types_2_score = [nn.Conv2d,nn.Linear],loss_f = sum_abs_loss, apply_final_mask = True, min_max=False,use_weight_mask=True):    #progressive skeletonization
	'''
	TO DO: This does not currently work with structured pruning, when target
	is a linear layer.
	use_weight_mask will allow for 'reviving' masked weights
	'''



	assert structure in ('weights','kernels','filters')

	device = next(model.parameters()).device  
	

	_ = model.eval()

	
	setup_net_for_mask(model)
	layers = OrderedDict([*model.named_modules()])


	#before getting the schedule of sparsities well get the total
	#parameters into the target by running the scoring function once

	scores = snip_score(model,dataloader,target_layer_name,unit,layer_types_2_score = layer_types_2_score, loss_f = loss_f, use_weight_mask=use_weight_mask)
	if structure in ['kernels','filters']:
		structured_scores = structure_scores(scores, model, structure=structure)
	else:
		structured_scores = scores

	if min_max:
		structured_scores = minmax_norm_scores(structured_scores)

	#total params
	# total_params = 0
	# for layer_name, layer_scores in structured_scores.items():
	# 	if layer_name == target_layer_name:
	# 		#EDIT, this might not be general in cases like branching models
	# 		#only weights leading into feature targets are important
	# 		total_params += params_2_target_in_layer(unit,layers[layer_name])
	# 	else:
	# 		total_params += torch.numel(layer_scores)

	total_params = params_2_target_from_scores(structured_scores,unit,target_layer_name,model)
	
	if num_params_to_keep is None:
		num_params_to_keep = ceil(keep_ratio*total_params)
	else:
		keep_ratio = num_params_to_keep/total_params       #num_params_to_keep arg overrides keep_ratio
	
	print('pruning %s'%structure)
	print('total parameters: %s'%str(total_params))
	print('parameters after pruning: %s'%str(num_params_to_keep))
	print('keep ratio: %s'%str(keep_ratio))
  
	if num_params_to_keep >= total_params:
		print('num params to keep > total params, no pruning to do')
		return

	
	#iteratively apply mask and score
	print("Pruning with %s pruning steps"%str(T))
	for t in range(1,T+1):
		
		print('step %s'%str(t))
		
		k = ceil(exp(t/T*log(num_params_to_keep)+(1-t/T)*log(total_params))) #exponential schedulr
		 
		print('%s params'%str(k))

		#mask model
		mask = mask_from_scores(structured_scores,num_params_to_keep=k)
		apply_mask(model,mask,zero_absent=False)

		#SNIP
		scores = snip_score(model,dataloader,target_layer_name,unit,layer_types_2_score = layer_types_2_score, loss_f = loss_f, use_weight_mask=use_weight_mask)
		if structure in ['kernels','filters']:
			structured_scores = structure_scores(scores, model, structure=structure)
		else:
			structured_scores = scores
		
		if min_max:
			structured_scores = minmax_norm_scores(structured_scores)

	#do we alter the final model to have the mask 
	# prescribed by FORCE, or keep it unmasked?
	if apply_final_mask:
		'applying final mask to model'
		mask = mask_from_scores(structured_scores,num_params_to_keep=k)
		apply_mask(model,mask,zero_absent=False)

		#print about final mask
		mask_ones = 0
		for layer_name,layer_mask in mask.items():
			mask_ones += int(torch.sum(layer_mask))
		print('final mask: %s/%s params (%s)'%(mask_ones,total_params,mask_ones/total_params))
	else:
		'keeping model unmasked'
		setup_net_for_mask(model) #sets mask to all 1s


	return structured_scores


def actgrad_kernel_score(model,dataloader,target_layer_name,unit,loss_f = sum_abs_loss,run_dissect_model=True):

	_ = model.eval()
	device = next(model.parameters()).device 

	if run_dissect_model:
		dis_model = dissect_model(deepcopy(model))
		model.to('cpu') #we need as much memory as we can get
	else:
		dis_model = model
	_ = dis_model.to(device).eval()


	all_layers = OrderedDict([*dis_model.named_modules()])
	dissected_layers = OrderedDict()

	for layer_name, layer in all_layers.items():
		if isinstance(layer,dissected_Conv2d):
			dissected_layers[layer_name] = layer

	#target_saver = feature_target_saver(model,layer_name,unit)
	scores = OrderedDict()
	with feature_target_saver(dis_model,target_layer_name,unit) as target_saver:
		for i, data in enumerate(dataloader, 0):
			#print('batch: '+str(i))
			inputs, label = data
			inputs = inputs.to(device)
			#label = label.to(device)

			dis_model.zero_grad() #very import!
			target_activations = target_saver(inputs)

			#feature collapse
			loss = loss_f(target_activations)
			loss.backward()

			#get weight-wise scores
			for layer_name,layer in dissected_layers.items():

				if layer.kernel_scores is None:
					if layer_name in scores.keys():
						raise Exception('kernel scores for %s not stored for batch %s'%(layer_name,str(i)))
					else:
						continue


				layer_scores = layer.kernel_scores
				
				if layer_name not in scores.keys():
					scores[layer_name] = layer_scores
				else:
					scores[layer_name] += layer_scores

	#reshape scores to in-out dimensions
	flattened_scores = OrderedDict()
	for layer_name, score in scores.items():
		flattened_scores[layer_name] = dissected_layers[layer_name].unflatten_kernel_scores( scores = scores[layer_name])

	del scores
	scores = flattened_scores


					
	# # eliminate layers with stored but all zero scores
	remove_keys = []
	for layer in scores:
		if torch.sum(scores[layer]) == 0.:
			remove_keys.append(layer)
	if len(remove_keys) > 0: 
		print('removing layers from scores with scores all 0:')
		for k in remove_keys:
			print(k)
			del scores[k]


	for layer_name, layer in dissected_layers.items():
		layer.kernel_scores = None

	if dissect_model:
		del dis_model #might be redundant
		model.to(device)

	return scores

class actgrad_filter_extractor(nn.Module):
	def __init__(self, model: nn.Module, layers: Iterable[str],absolute=True):
		super().__init__()
		self.model = model
		self.layers = layers
		self.activations = {layer: None for layer in layers}
		self.gradients = {layer: None for layer in layers}
		self.absolute = absolute

	def __enter__(self, *args):
		#self.remove_all_hooks() 
		self.hooks = {'forward':{},
				 	  'backward':{}}   #saving hooks to variables lets us remove them later if we want
		for layer_id in self.layers:
			layer = dict([*self.model.named_modules()])[layer_id]
			self.hooks['forward'][layer_id] = layer.register_forward_hook(self.save_activations(layer_id)) #execute on forward pass
			self.hooks['backward'][layer_id] = layer.register_backward_hook(self.save_gradients(layer_id))    #execute on backwards pass      
		return self

	def __exit__(self, *args): 
		self.remove_all_hooks()


	def save_activations(self, layer_id: str) -> Callable:
		def fn(module, input, output):  #register_hook expects to recieve a function with arguments like this
			#output is what is return by the layer with dim (batch_dim x out_dim), sum across the batch dim
			if self.absolute:
				batch_summed_output = torch.sum(torch.abs(output),dim=0).detach().cpu()
			else:
				batch_summed_output = torch.sum(output,dim=0).detach().cpu()
			if self.activations[layer_id] is None:
				self.activations[layer_id] = batch_summed_output
			else:
				self.activations[layer_id] +=  batch_summed_output
		return fn
	
	def save_gradients(self, layer_id: str) -> Callable:
		def fn(module, grad_input, grad_output):
			if self.absolute:
				batch_summed_output = torch.sum(torch.abs(grad_output[0]),dim=0).detach().cpu() #grad_output is a tuple with 'device' as second item
			else:
				batch_summed_output = torch.sum(grad_output[0],dim=0).detach().cpu()

			if self.gradients[layer_id] is None:
				self.gradients[layer_id] = batch_summed_output
			else:
				self.gradients[layer_id] +=  batch_summed_output 
		return fn
	
	def remove_all_hooks(self):
		for layer_id in self.layers:
			self.hooks['forward'][layer_id].remove()
			self.hooks['backward'][layer_id].remove()


def actgrad_filter_score(model,dataloader,target_layer_name,unit,loss_f=sum_abs_loss, absolute=True,return_target=False,relu=True,score_type = 'actgrad'):
    all_layers = OrderedDict([*model.named_modules()])
    scoring_layers = []
    for layer in all_layers:
        if layer == target_layer_name:   #HACK MIGHT NOT WORK WITH INCEPTION
            break
        if isinstance(all_layers[layer],torch.nn.modules.conv.Conv2d):
            scoring_layers.append(layer)
            
    _ = model.eval()
    device = next(model.parameters()).device 
    
    scores = OrderedDict()
    
    
    overall_loss = 0
    with feature_target_saver(model,target_layer_name,unit) as target_saver:
        with actgrad_filter_extractor(model,scoring_layers,absolute = absolute) as score_saver:
            for i, data in enumerate(dataloader, 0):
                inputs, label = data
                inputs = inputs.to(device)

                model.zero_grad() #very import!
                target_activations = target_saver(inputs)

                #feature collapse
                loss = loss_f(target_activations)
                overall_loss+=loss
                loss.backward()

            #get average by dividing result by length of dset
            activations = score_saver.activations
            gradients = score_saver.gradients

            for l in scoring_layers:
                
                if score_type == 'gradients':
                    layer_scores = gradients[l]
                elif score_type == 'activations':
                    #get mask where gradient zero (outside receptive field)
                    grad_mask = (gradients[l] != 0.).float()
                    if relu:
                        rl=nn.ReLU()
                        layer_scores = (rl(activations[l]) * grad_mask).mean(dim=(1,2))
                    else:
                        layer_scores = (activations[l] * grad_mask).mean(dim=(1,2))
                        
                else:   
                    if relu:
                        rl=nn.ReLU()
                        layer_scores = (rl(activations[l]) * gradients[l]).mean(dim=(1,2))
                    else:
                        layer_scores = (activations[l] * gradients[l]).mean(dim=(1,2))
                    
                    
                if l not in scores.keys():
                    scores[l] = layer_scores
                else:
                    scores[l] += layer_scores


    remove_keys = []
    for layer in scores:
        if torch.sum(scores[layer]) == 0.:
            remove_keys.append(layer)
    if len(remove_keys) > 0: 
        print('removing layers from scores with scores all 0:')
        for k in remove_keys:
            print(k)
            del scores[k]


    model.zero_grad() 
    if return_target:
        return scores,float(overall_loss.detach().cpu())
    else:
        return scores



def magnitude_scores_from_scores(scores,model,target_layer_name,unit,structure='kernels'):

	'''
	This function presumes you already have a scores file, and uses that
	to make a magnitude scores file based on weights of model.
	It uses a scores file (from 'snip' for example), so that it
	can disregard weights dissconnected from the target
	'''
	_ = model.eval()
	model_layers = OrderedDict([*model.named_modules()])
	
	mag_scores = OrderedDict()

	for layer_name, layer_scores in scores.items():
		layer = model_layers[layer_name]
		if layer_name != target_layer_name:
			w = layer.weight.detach().cpu()
			w_abs = torch.abs(w)
			mag_scores[layer_name] = w_abs 
		else: #EDIT, only for basis units
			w_empty = torch.zeros(layer.weight.shape).cpu()
			w_unit = torch.abs(layer.weight[unit].detach().cpu())
			w_empty[unit] = w_unit
			mag_scores[layer_name] = w_empty
		if structure == 'kernels':
			mag_scores[layer_name] = mag_scores[layer_name].mean(dim=(2,3))
		elif structure == 'filters':
			mag_scores[layer_name] = mag_scores[layer_name].mean(dim=(1,2,3))

	return mag_scores


def random_scores_from_scores(scores,target_layer_name,unit):

	random_scores = OrderedDict()

	for layer_name, layer_scores in scores.items():
		if layer_name != target_layer_name:
			random_scores[layer_name] = torch.abs(torch.rand(layer_scores.shape))
		else: #EDIT, only for basis units
			scores_empty = torch.zeros(layer_scores.shape).cpu()
			scores_empty[unit] = torch.abs(torch.rand(layer_scores[unit].shape))
			random_scores[layer_name] = scores_empty

	return random_scores


#### Score manipulations #####
'''
functions for manipulating scores
'''

def structure_scores(scores, model, structure='kernels'):
	
	assert structure in ['kernels','filters']
	layers = OrderedDict([*model.named_modules()])
	
	if structure == 'kernels':
		collapse_dims = (2,3)
	else:
		collapse_dims = (1,2,3)
		
	structured_scores = OrderedDict()
	for layer_name in scores:
		if isinstance(layers[layer_name],nn.Conv2d):
			structured_scores[layer_name] = torch.mean(scores[layer_name],dim=collapse_dims)
			
	return structured_scores      
		

def minmax_norm_scores(scores, min=0, max=1):
	out_scores = OrderedDict()
	for layer_name, scores in scores.items():
		old_min = torch.min(scores)
		old_max = torch.max(scores)
		out_scores[layer_name] = (scores - old_min)/(old_max - old_min)*(max - min) + min

	return out_scores



def get_num_params_from_cum_score(scores,cum_score,tolerance=.005):
	'''
	Given scores, and a target cumulative score (between 0-1)
	This function will return the number of params to keep in the
	mask to achieve that cumulative score, and the sparsity level 
	at that cumulative score
	'''


	all_scores = []
	for l in scores:
		all_scores.append(scores[l].flatten())
	all_scores = torch.cat(all_scores).flatten()
	sort_scores = torch.sort(all_scores,descending=True).values
	total_sum = all_scores.sum()
	target_sum = total_sum*cum_score
	tolerance = .001
	l_bound = target_sum-target_sum*tolerance
	u_bound = target_sum+target_sum*tolerance

	#binary search
	li=0
	ui=len(sort_scores)
	i = int(len(sort_scores)/2)
	current_sum = sort_scores[0:i].sum()
	while not l_bound<=current_sum<=u_bound:
		if current_sum<l_bound:
			li = i
			i = li+int((ui-li)/2)
		else:
			ui = i
			i = ui - int((ui-li)/2)
		current_sum = sort_scores[0:i].sum()

	return i

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
from copy import deepcopy


#### MASKS #####
'''
Functions for masking the network, given scores
'''

def mask_from_scores(scores, sparsity=None,num_params_to_keep=None,model = None,unit=None,target_layer=None,relevant_sparsity=True):
	'''
	relevant sparsity: the sparsity given is with respect to 'relevant' parameters
	'''
	assert not ((sparsity is None) and (num_params_to_keep is None))

	keep_masks = OrderedDict()
	
	#flatten
	scores_flat = torch.cat([torch.flatten(x) for x in scores.values()])
	norm_factor = torch.sum(abs(scores_flat))
	scores_flat.div_(norm_factor)

	#num kept params
	if not num_params_to_keep is None:
		k = num_params_to_keep
	elif relevant_sparsity:
		assert not (model is None or unit is None or target_layer is None)
		total_params = params_2_target_from_scores(scores,unit,target_layer,model)
		k = int(total_params * sparsity)
	else:
		total_params = len(scores_flat)
		k = int(total_params * sparsity)

	#get threshold score
	threshold, _ = torch.topk(scores_flat, k, sorted=True)
	acceptable_score = threshold[-1]


	
	if acceptable_score == 0:
		print('gradients from this feature are sparse, the minimum acceptable score at this sparsity has a score of zero! we will return a mask thats smaller than you asked, by masking all parameters with a score of zero.')

	for layer_name in scores:
		layer_scores = scores[layer_name]
		keep_masks[layer_name] = (layer_scores / norm_factor > acceptable_score).float()
	
	return keep_masks


def masked_conv2d_forward(self, x):

	#pass input through conv and weight mask

	x = F.conv2d(x, self.weight * self.weight_mask, self.bias,
					self.stride, self.padding, self.dilation, self.groups) 

	return x

def masked_linear_forward(self, x):

	x = F.linear(x, self.weight * self.weight_mask, self.bias)

	return x

def setup_net_for_mask(model):

	#same naming trick as before 
	layers = OrderedDict([*model.named_modules()])

	for layer in layers.values():
		if isinstance(layer, nn.Conv2d):
			layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
			layer.forward = types.MethodType(masked_conv2d_forward, layer)
		elif isinstance(layer, nn.Linear):
			layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
			layer.forward = types.MethodType(masked_linear_forward, layer)
			
def apply_mask(model,mask, zero_absent=True):

	layers = OrderedDict([*model.named_modules()])
	setup_net_for_mask(model)

	#mask may be structured, lets 'expand' it before applying it to the model
	expanded_mask = expand_structured_mask(mask,model)

	for layer_name in expanded_mask:
		layers[layer_name].weight_mask = nn.Parameter(expanded_mask[layer_name].to(layers[layer_name].weight.device))
	if zero_absent:
		#mask all layers not specified in the mask
		for layer_name in layers:
			if layer_name not in expanded_mask.keys():
				try:
					layers[layer_name].weight_mask = nn.Parameter(torch.zeros_like(layers[layer_name].weight))
				except:
					pass



def expand_structured_mask(mask,model):
	'''Structured mask might have shape (filter, channel) for kernel structured mask, but the weights have
		shape (filter,channel,height,width), so we make a new weight wise mask based on the structured mask'''

	layers = OrderedDict([*model.named_modules()])
	expanded_mask = OrderedDict()

	for layer_name, layer_mask in mask.items():
		w = layers[layer_name].weight
		m = deepcopy(layer_mask)
		while len(m.shape) < len(w.shape):
			m = m.unsqueeze(dim=-1)
		m = m.expand(w.shape)
		expanded_mask[layer_name] = m
	
	return expanded_mask


def structured_mask_from_mask(mask, structure = 'kernels'):
	
	if isinstance(mask,dict):
		layer_keys = list(mask.keys())
		mask_list = []
		for i in mask:
			mask_list.append(mask[i])
		mask = mask_list

	if structure == 'weights':
		raise ValueError("to create a weight mask use the function circuit_pruner.force.expand_structured_mask")
	if structure not in ['kernels','edges','filters','nodes']:
		raise ValueError("Argument 'structure' must be in ['weights','kernels','edges','filters','nodes']")



	if len(mask[0].shape) == 4:
		in_structure = 'weights'
	elif len(mask[0].shape) == 2:
		in_structure = 'kernels'
	elif len(mask[0].shape) == 1:
		in_structure = 'filters'
	else:
		raise ValueError("Dont understand Shape %s of input mask, must be 1,2 or 4 (filters,kernels,weights)"%str(len(mask[0].shape)))

	if in_structure == structure:
		print('provided mask already of structure %s'%structure)
		return mask

	out_mask = []

	for m in mask:
		if structure in ['filters','nodes']:
			m_flat = torch.reshape(m,(m.shape[0],-1))
			z = torch.zeros(m_flat.shape[1])
			m_out = ~torch.all(m_flat==z,dim=1)

		else:
			m_flat = torch.reshape(m,(m.shape[0]*m.shape[1],-1))
			z = torch.zeros(m_flat.shape[1])
			m_out = ~torch.reshape(torch.all(m_flat==z,dim=1),(m.shape[0],m.shape[1]))

		m_out= m_out.type(torch.FloatTensor)
		out_mask.append(m_out)


	out_mask_dict = OrderedDict()
	for i in range(len(out_mask)):
		out_mask_dict[layer_keys[i]] = out_mask[i]

	return out_mask




def mask_intersect_over_union(mask1,mask2):
	iou = {}
	for layer_name in mask1:
		try:
			intersect_mask = mask1[layer_name]*mask2[layer_name]
			union_mask = torch.ceil((mask1[layer_name]+mask2[layer_name])/2)
			iou[layer_name] = (torch.sum(intersect_mask)/torch.sum(union_mask))
		except:
			print('skipping %s'%layer_name)
			continue
	return iou


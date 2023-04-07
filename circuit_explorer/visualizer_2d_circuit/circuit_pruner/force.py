from collections import OrderedDict
from circuit_pruner.custom_exceptions import TargetReached
from circuit_pruner.utils import *
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import types

from math import log, exp, ceil

	
def get_last_layer_from_feature_targets(net, feature_targets):
	
	
	#get dict version of feature targets
	feature_targets = feature_targets_list_2_dict(feature_targets)
	target_layers =  feature_targets.keys()      
	
	
	def check_layers(net,last_layer=None):

		if last_layer is not None:
			return last_layer

		if hasattr(net, "_modules"):
			for name, layer in reversed(net._modules.items()):  #use reversed to start from the end of the network

				if layer is None:
					# e.g. GoogLeNet's aux1 and aux2 layers
					continue

				if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
					if layer.ref_name in target_layers:
						last_layer = layer.ref_name
						break
		
						
				last_layer = check_layers(layer, last_layer=last_layer)
		
		return last_layer

		   
	last_layer = check_layers(net)    

	if last_layer is None:
		print('No effective last layer found!')
	else:
		print('%s effective last layer'%last_layer)
		
	return last_layer
	

 
def feature_targets_list_2_dict(feature_targets,feature_targets_coefficients=None):
	if isinstance(feature_targets_coefficients,list):
		feature_targets_coefficients_ls = feature_targets_coefficients
		feature_targets_coefficients = {}
	
	#get dict version of feature targets
	if isinstance(feature_targets,list):
		feature_targets_ls = feature_targets
		feature_targets = {}
		for i,feature_conj in enumerate(feature_targets_ls):
			layer, feature = feature_conj.split(':')
			if layer.strip() in feature_targets:
				feature_targets[layer.strip()].append(int(feature.strip()))
			else:
				feature_targets[layer.strip()] = [int(feature.strip())]
			if feature_targets_coefficients is not None:
				if layer.strip() in feature_targets_coefficients:
					feature_targets_coefficients[layer.strip()].append(feature_targets_coefficients_ls[i])
				else:
					feature_targets_coefficients[layer.strip()] = [feature_targets_coefficients_ls[i]]
				
	assert isinstance(feature_targets,dict)
	
	if feature_targets_coefficients is None:
		return feature_targets
	else:
		return feature_targets, feature_targets_coefficients
	


def setup_net_for_circuit_prune(net, feature_targets=None,save_target_activations=False,rank_field = 'image'):
	
	if not isinstance(rank_field,list):
		assert rank_field in ('image','min','max')

	#name model modules
	ref_name_modules(net)
	
	
	last_layer = None
	#get dict version of feature targets
	if feature_targets is not None:
		feature_targets = feature_targets_list_2_dict(feature_targets)
	
		#get effective last_layer
		last_layer = get_last_layer_from_feature_targets(net, feature_targets)
		
	
	def setup_layers(net):
		if hasattr(net, "_modules"):
			for name, layer in net._modules.items():

				if layer is None:
					# e.g. GoogLeNet's aux1 and aux2 layers
					continue

				if isinstance(layer, nn.Conv2d):
					layer.filter_mask = None
					layer.filter_mask_expanded = None
					layer.save_target_activations = save_target_activations
					layer.target_activations = {}
					
					layer.last_layer = False
					if layer.ref_name == last_layer:
						layer.last_layer = True
						
					
					layer.rank_field = rank_field
					layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
					
					layer.feature_targets_indices = None
					if feature_targets is not None:
						if layer.ref_name in feature_targets: #layer has feature targets in it
							layer.feature_targets_indices = feature_targets[layer.ref_name]

					#setup masked forward pass
					layer.forward = types.MethodType(circuit_prune_forward_conv2d, layer)

				elif isinstance(layer, nn.Linear):
					layer.save_target_activations = save_target_activations
					layer.target_activations = {}
					
					layer.last_layer = False
					if layer.ref_name == last_layer:
						layer.last_layer = True
					
					layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
					
					layer.feature_targets_indices = None
					if feature_targets is not None:
						if layer.ref_name in feature_targets: #layer has feature targets in it
							layer.feature_targets_indices = feature_targets[layer.ref_name]

					#setup masked forward pass
					layer.forward = types.MethodType(circuit_prune_forward_linear, layer)


				setup_layers(layer)

		   
	setup_layers(net)
	

	
#Error classes for breaking forward pass of model
# define Python user-defined exceptions
'''
class ModelBreak(Exception):
	"""Base class for other exceptions"""
	pass

class TargetReached(ModelBreak):
	"""Raised when the output target for a subgraph is reached, so the model doesnt neeed to be run forward any farther"""
	pass    
'''

	
def circuit_prune_forward_conv2d(self, x):
		
	#pass input through conv and weight mask

	x = F.conv2d(x, self.weight * self.weight_mask, self.bias,
					self.stride, self.padding, self.dilation, self.groups) 

#     x = F.conv2d(x, self.weight, self.bias,
#                     self.stride, self.padding, self.dilation, self.groups) 
	if self.filter_mask is not None:
		if self.filter_mask_expanded is None:
			self.filter_mask_expanded = nn.Parameter(self.filter_mask(x.shape[0],x.shape[2],x.shape[3],x.shape[1]).permute(0,3,1,2).to(x.device))
			self.filter_mask_expanded.requires_grad = False
		x = x*self.filter_mask_expanded

	

	self.target_activations = {}
	
	#gather feature targets
	if self.feature_targets_indices is not None: #there are feature targets in the conv 
		self.feature_targets = {}

		for feature_idx in self.feature_targets_indices:
			if self.rank_field == 'image':
				avg_activations = x.mean(dim=(0, 2, 3))
				self.feature_targets[feature_idx] = avg_activations[feature_idx]
				
			elif self.rank_field == 'max':
				max_acts = x.view(x.size(0),x.size(1), x.size(2)*x.size(3)).max(dim=-1).values
				max_acts_target = max_acts[:,feature_idx]
				self.feature_targets[feature_idx] = max_acts_target.mean()
				
			elif self.rank_field == 'min':
				min_acts = x.view(x.size(0),x.size(1), x.size(2)*x.size(3)).min(dim=-1).values
				min_acts_target = min_acts[:,feature_idx]
				self.feature_targets[feature_idx] = min_acts_target.mean()

			elif isinstance(self.rank_field,list):
				if isinstance(self.rank_field[0],list):
					d = x.get_device()
					if d == -1:
						device='cpu'
					else:
						device='cuda:'+str(d)
					act_targets_sum = torch.FloatTensor(1).zero_().to(device) 
					for i in range(len(self.rank_field)):
						act_targets_sum += x[i,feature_idx,int(self.rank_field[i][0]),int(self.rank_field[i][1])]
				else:
					act_targets_sum = x[:,feature_idx,int(self.rank_field[0]),int(self.rank_field[1])].sum()
					
				self.feature_targets[feature_idx] = act_targets_sum/x.shape[0]


			if self.save_target_activations:
				self.target_activations[feature_idx] = x[:,feature_idx,:,:].data.to('cpu')
			
	if self.last_layer: #stop model forward pass if all targets reached
		raise TargetReached
	
	return x
	



def circuit_prune_forward_linear(self, x):
	
	#pass input through weights and weight mask
	x = F.linear(x, self.weight * self.weight_mask, self.bias)
#     x = F.linear(x, self.weight, self.bias)
	
	self.target_activations = {}
	
	#gather feature targets
	if self.feature_targets_indices is not None: #there are feature targets in the conv 
		self.feature_targets = {}

		for feature_idx in self.feature_targets_indices:
			avg_activations = x.mean(dim=(0))
			self.feature_targets[feature_idx] = avg_activations[feature_idx] 
			
			if self.save_target_activations:
				self.target_activations[feature_idx] = x[:,feature_idx].data.to('cpu')
	
	if self.last_layer: #stop model forward pass if all targets reached
		raise TargetReached
	
	return x
  
 
def get_feature_target_from_net(net, feature_targets, feature_targets_coefficients,device=None):
	
	if device is None:
		device = 'cuda' if torch.cuda.is_available() else 'cpu'
	
	if feature_targets_coefficients is None:
		feature_targets_indices = feature_targets_list_2_dict(feature_targets)
	else:
		feature_targets_indices,feature_targets_coefficients = feature_targets_list_2_dict(feature_targets,feature_targets_coefficients=feature_targets_coefficients) 
	

	
	def fetch_targets_values(net,feature_targets_values = {}):
		if hasattr(net, "_modules"):
			for name, layer in net._modules.items():

				if layer is None:
					# e.g. GoogLeNet's aux1 and aux2 layers
					continue

				if layer.ref_name in feature_targets_indices.keys():
					feature_targets_values[layer.ref_name] = []
					for idx in layer.feature_targets:
						feature_targets_values[layer.ref_name].append(layer.feature_targets[idx])
						
				feature_targets_values = fetch_targets_values(layer, feature_targets_values = feature_targets_values)
		
		return feature_targets_values
				
	feature_targets_values = fetch_targets_values(net)
	
	target = None
	for layer in feature_targets_values:
		for idx in range(len(feature_targets_values[layer])):
			coeff = 1
			if feature_targets_coefficients is not None:
				coeff = feature_targets_coefficients[layer][idx] 
			
			if target is None:
				target = coeff*feature_targets_values[layer][idx]
			else:
				target += coeff*feature_targets_values[layer][idx]

	return target



def clear_feature_targets_from_net(net):

	
	def clear_layers(net):
		if hasattr(net, "_modules"):
			for name, layer in net._modules.items():

				if layer is None:
					# e.g. GoogLeNet's aux1 and aux2 layers
					continue

				if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
					layer.feature_targets = None


				clear_layers(layer)

		   
	clear_layers(net)
	

def reset_masks_in_net(net):


	def reset_layers(net):
		if hasattr(net, "_modules"):
			for name, layer in net._modules.items():

				if layer is None:
					# e.g. GoogLeNet's aux1 and aux2 layers
					continue

				if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
					layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))


				reset_layers(layer)


	reset_layers(net)
	
	
def save_target_activations_in_net(net,save=True):
	
	def reset_layers(net):
		if hasattr(net, "_modules"):
			for name, layer in net._modules.items():

				if layer is None:
					# e.g. GoogLeNet's aux1 and aux2 layers
					continue

				if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
					layer.save_target_activations = save


				reset_layers(layer)


	reset_layers(net)
   


def get_saved_target_activations_from_net(net,detach=True):

	def fetch_activations(net,target_activations = {},detach=detach):
		if hasattr(net, "_modules"):
			for name, layer in net._modules.items():

				if layer is None:
					# e.g. GoogLeNet's aux1 and aux2 layers
					continue

				if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
					if layer.save_target_activations:
						if layer.target_activations != {}:
							for idx in layer.target_activations:
								if detach:
									target_activations[layer.ref_name+':'+str(idx)] = layer.target_activations[idx].cpu().detach().numpy().astype('float32') 
								else:
									target_activations[layer.ref_name+':'+str(idx)] = layer.target_activations[idx]


						
				target_activations = fetch_activations(layer, target_activations = target_activations,detach=detach)
		
		return target_activations
				
	target_activations = fetch_activations(net)
	

	return target_activations


def apply_filter_mask(net,mask):
	count = 0
	for layer in net.modules():
		if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
			if mask is not None and count < len(mask): #we have a mask for these weights 
				layer.filter_mask = mask[count]
			count += 1


def apply_mask(net,mask):
	count = 0
	for layer in net.modules():
		if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
			if mask is not None and count < len(mask): #we have a mask for these weights 
				layer.weight_mask = nn.Parameter(mask[count].to(layer.weight.device))
			else:
				layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight).to(layer.weight.device))
			#nn.init.xavier_normal_(layer.weight)
			layer.weight.requires_grad = False
			count += 1


def make_net_mask_only(net):
	#function keeps net mask but gets rid of other bells and whistles
	for layer in net.modules():
		if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
			layer.save_target_activations = False
			layer.target_activations = {}
			layer.last_layer = False
			layer.feature_targets_indices = None

	
def circuit_SNIP(net, dataloader, feature_targets = None, feature_targets_coefficients = None, full_dataset = True, keep_ratio=.1, num_params_to_keep=None, device=None, structure='weights', mask=None, criterion= None, setup_net=True,rank_field='image', use_abs_ranks=True,return_ranks = False):
	'''
	if num_params_to_keep is specified, this argument overrides keep_ratio
	'''

	assert structure in ('weights','kernels','filters')    
	
	if device is None:
		device = 'cuda' if torch.cuda.is_available() else 'cpu'
  

	#set up cirterion
	if criterion is None:
		criterion = torch.nn.CrossEntropyLoss()



	if setup_net:
		setup_net_for_circuit_prune(net, feature_targets=feature_targets, rank_field = rank_field)
		

	
	#apply current mask
	count = 0
	for layer in net.modules():
		if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
			if mask is not None and count < len(mask): #we have a mask for these weights 
				layer.weight_mask = nn.Parameter(mask[count])
			else:
				layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
			#nn.init.xavier_normal_(layer.weight)
			layer.weight.requires_grad = False
			count += 1

	

	
	#do we iterate through the whole dataset or not
	iter_dataloader = iter(dataloader)
	
	iters = 1
	if full_dataset:
		iters = len(iter_dataloader)
	
	
	grads = [] #computed scores
	

	for it in range(iters):
		clear_feature_targets_from_net(net)
		
		# Grab a single batch from the training dataset
		inputs, targets = next(iter_dataloader)
		inputs = inputs.to(device)
		targets = targets.to(device)




		# Compute gradients (but don't apply them)
		net.zero_grad()
		
		#Run model forward until all targets reached
		try:
			outputs = net.forward(inputs)
		except TargetReached:
			pass
		
		#get proper loss
		if feature_targets is None:
			loss = criterion(outputs, targets)
		else:   #the real target is feature values in the network
			loss = get_feature_target_from_net(net, feature_targets, feature_targets_coefficients,device=device)
		
		loss.backward()

		#get weight-wise scores
		if grads == []:
			for layer in net.modules():
				if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
					if use_abs_ranks:
						grads.append(torch.abs(layer.weight_mask.grad))
					else:
						grads.append(layer.weight_mask.grad)
					if layer.last_layer:
						break
		else:
			count = 0
			for layer in net.modules():
				if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
					grads[count] += torch.abs(layer.weight_mask.grad)       
					count += 1
					if layer.last_layer:
						break
	  

				
	#structure scoring by weights, kernels, or filters   
	
	if structure == 'weights':
		structure_grads = grads
	elif structure == 'kernels':
		structure_grads = []
		for grad in grads:
			if len(grad.shape) == 4: #conv2d layer
				structure_grads.append(torch.mean(grad,dim = (2,3))) #average across height and width of each kernel
	else:
		structure_grads = []
		for grad in grads:
			if len(grad.shape) == 4: #conv2d layer
				structure_grads.append(torch.mean(grad,dim = (1,2,3))) #average across channel height and width of each filter
		

	# Gather all scores in a single vector and normalise
	all_scores = torch.cat([torch.flatten(x) for x in structure_grads])
	norm_factor = torch.sum(abs(all_scores))
	all_scores.div_(norm_factor)
	
	#get num params to keep
	if num_params_to_keep is None:
		num_params_to_keep = int(len(all_scores) * keep_ratio)
	threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
	acceptable_score = threshold[-1]


	keep_masks = []
	if acceptable_score == 0:
		print('gradients from this feature are sparse,\
the minimum acceptable rank at this sparsity has a score of zero! \
we will return a mask thats smaller than you asked, by masking all \
parameters with a score of zero.')
	
		for g in structure_grads:
			keep_masks.append((g / norm_factor > acceptable_score).float())
	else:
		for g in structure_grads:
			keep_masks.append(((g / norm_factor) > acceptable_score).float())

	#print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks])))

	if not return_ranks:
		return keep_masks
	else:
		return grads,keep_masks

	


def circuit_snip_rank(net, dataloader, feature_targets = None, feature_targets_coefficients = None, full_dataset = True, device=None, mask=None, criterion= None, setup_net=True,rank_field='image'):

	
	if device is None:
		device = 'cuda' if torch.cuda.is_available() else 'cpu'
  

	#set up cirterion
	if criterion is None:
		criterion = torch.nn.CrossEntropyLoss()



	if setup_net:
		setup_net_for_circuit_prune(net, feature_targets=feature_targets, rank_field = rank_field)
		

	
	#apply current mask
	count = 0
	for layer in net.modules():
		if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
			if mask is not None and count < len(mask): #we have a mask for these weights 
				layer.weight_mask = nn.Parameter(mask[count])
			else:
				layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
			#nn.init.xavier_normal_(layer.weight)
			layer.weight.requires_grad = False
			count += 1

	

	
	#do we iterate through the whole dataset or not
	iter_dataloader = iter(dataloader)
	
	iters = 1
	if full_dataset:
		iters = len(iter_dataloader)
	
	
	grads = [] #computed scores
	

	for it in range(iters):
		clear_feature_targets_from_net(net)
		
		# Grab a single batch from the training dataset
		inputs, targets = next(iter_dataloader)
		inputs = inputs.to(device)
		targets = targets.to(device)


		# Compute gradients (but don't apply them)
		net.zero_grad()
		
		#Run model forward until all targets reached
		try:
			outputs = net.forward(inputs)
		except TargetReached:
			pass
		
		#get proper loss
		if feature_targets is None:
			loss = criterion(outputs, targets)
		else:   #the real target is feature values in the network
			loss = get_feature_target_from_net(net, feature_targets, feature_targets_coefficients,device=device)
		
		loss.backward()

		#get weight-wise scores
		if grads == []:
			for layer in net.modules():
				if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
					try:
						grads.append(torch.abs(layer.weight_mask.grad))
					except:
						grads.append(torch.zeros(layer.weight_mask.shape))
						#import pdb; pdb.set_trace()
					if layer.last_layer:
						break
		else:
			count = 0
			for layer in net.modules():
				if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
					try:
						grads[count] += torch.abs(layer.weight_mask.grad) 
					except:
						continue       
					count += 1
					if layer.last_layer:
						break
	  
	return(grads)





	
# def apply_prune_mask(net, keep_masks):

#     # Before I can zip() layers and pruning masks I need to make sure they match
#     # one-to-one by removing all the irrelevant modules:
#     prunable_layers = filter(
#         lambda layer: isinstance(layer, nn.Conv2d) or isinstance(
#             layer, nn.Linear), net.modules())

#     for layer, keep_mask in zip(prunable_layers, keep_masks):
#         assert (layer.weight.shape == keep_mask.shape)

#         def hook_factory(keep_mask):
#             """
#             The hook function can't be defined directly here because of Python's
#             late binding which would result in all hooks getting the very last
#             mask! Getting it through another function forces early binding.
#             """

#             def hook(grads):
#                 return grads * keep_mask

#             return hook

#         # mask[i] == 0 --> Prune parameter
#         # mask[i] == 1 --> Keep parameter

#         # Step 1: Set the masked weights to zero (NB the biases are ignored)
#         # Step 2: Make sure their gradients remain zero
#         layer.weight.data[keep_mask == 0.] = 0.
#         layer.weight.register_hook(hook_factory(keep_mask))
		
		
def expand_structured_mask(mask,net):
	'''Structured mask might have shape (filter, channel) for kernel structured mask, but the weights have
		shape (filter,channel,height,width), so we make a new weight wise mask based on the structured mask'''

	weight_mask = []
	count=0
	for layer in net.modules():
		if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
			if count < len(mask):
				weight_mask.append(mask[count])
				while len(weight_mask[-1].shape) < 4:
					weight_mask[-1] = weight_mask[-1].unsqueeze(dim=-1)
				weight_mask[-1] = weight_mask[-1].expand(layer.weight.shape)
			count+= 1
	return weight_mask





def circuit_FORCE_pruning(model, dataloader, feature_targets = None,feature_targets_coefficients = None,keep_ratio=.1, T=10, use_abs_ranks=True, full_dataset = True, num_params_to_keep=None, device=None, structure='kernels', rank_field = 'image', mask=None, setup_net=True, return_ranks = False):    #progressive skeletonization

	
	assert structure in ('weights','kernels','filters')
	assert rank_field in ('image','max','min')
	
	if device is None:
		device = 'cuda' if torch.cuda.is_available() else 'cpu'

	model.to('cpu').eval()
	net = deepcopy(model)
	net = net.to(device).eval()	
	for param in net.parameters():
		param.requires_grad = False
	
	if setup_net:
		setup_net_for_circuit_prune(net, feature_targets=feature_targets, rank_field = rank_field)
	
	
	#get total params given feature target might exclude some of network
	total_params = 0

	for layer in net.modules():
		if structure == 'weights' and (isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)):
			if not layer.last_layer:  #all params potentially important
				total_params += len(layer.weight.flatten())
			else:    #only weights leading into feature targets are important
				total_params += len(layer.feature_targets_indices)*int(layer.weight.shape[1])
				break
		elif isinstance(layer, nn.Conv2d):
			if not layer.last_layer:  #all params potentially important
				if structure == 'kernels':
					total_params += int(layer.weight.shape[0]*layer.weight.shape[1])
				else:
					total_params += int(layer.weight.shape[0])
					
			else: #only weights leading into feature targets are important
				if structure == 'kernels':
					total_params += int(len(layer.feature_targets_indices)*layer.weight.shape[1])
				else:
					total_params += len(layer.feature_targets_indices)
				
				break
	
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

	print("Pruning with %s pruning steps"%str(T))
	for t in range(1,T+1):
		
		print('step %s'%str(t))
		
		k = ceil(exp(t/T*log(num_params_to_keep)+(1-t/T)*log(total_params))) #exponential schedulr
		 
		print('%s params'%str(k))
		
		#SNIP
		if not return_ranks:
			struct_mask = circuit_SNIP(net, dataloader, num_params_to_keep=k, feature_targets = feature_targets, feature_targets_coefficients = feature_targets_coefficients, use_abs_ranks = use_abs_ranks, structure=structure, mask=mask, full_dataset = full_dataset, device=device,setup_net=False)
		else:
			grads,struct_mask = circuit_SNIP(net, dataloader, num_params_to_keep=k, feature_targets = feature_targets, feature_targets_coefficients = feature_targets_coefficients, use_abs_ranks = use_abs_ranks, structure=structure, mask=mask, full_dataset = full_dataset, device=device,setup_net=False,return_ranks=True)
		if structure is not 'weights':
			mask = expand_structured_mask(struct_mask,net) #this weight mask will get applied to the network on the next iteration
		else:
			mask = struct_mask

	apply_mask(net,mask)

	mask_total = 0
	mask_ones = 0
	for l in mask:
		mask_ones += int(torch.sum(l))
		mask_total += int(torch.numel(l))
	print('final mask: %s/%s params (%s)'%(mask_ones,mask_total,mask_ones/mask_total))


	if not return_ranks:
		return struct_mask
	else:
		return grads,struct_mask








def circuit_RECON_pruning(net, dataloader, feature_targets = None,feature_targets_coefficients = None, T=10,full_dataset = True, keep_ratio=.1, num_params_to_keep=None, device=None, structure='weights', rank_field = 'image', mask=None, setup_net=True):    #progressive skeletonization

	
	assert structure in ('weights','kernels','filters')
	assert rank_field in ('image','max','min')
	
	if device is None:
		device = 'cuda' if torch.cuda.is_available() else 'cpu'
	net = net.to(device)
	
	if setup_net:
		setup_net_for_circuit_prune(net, feature_targets=feature_targets, rank_field = rank_field)
	
	
	#get total params given feature target might exclude some of network
	total_params = 0

	for layer in net.modules():
		if structure == 'weights' and (isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)):
			if not layer.last_layer:  #all params potentially important
				total_params += len(layer.weight.flatten())
			else:    #only weights leading into feature targets are important
				total_params += len(layer.feature_targets_indices)*int(layer.weight.shape[1])
				break
		elif isinstance(layer, nn.Conv2d):
			if not layer.last_layer:  #all params potentially important
				if structure == 'kernels':
					total_params += int(layer.weight.shape[0]*layer.weight.shape[1])
				else:
					total_params += int(layer.weight.shape[0])
					
			else: #only weights leading into feature targets are important
				if structure == 'kernels':
					total_params += int(len(layer.feature_targets_indices)*layer.weight.shape[1])
				else:
					total_params += len(layer.feature_targets_indices)
				
				break
	
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

	print("Pruning with %s pruning steps"%str(T))
	for t in range(1,T+1):
		
		print('step %s'%str(t))
		
		k = ceil(exp(t/T*log(num_params_to_keep)+(1-t/T)*log(total_params))) #exponential schedulr
		 
		print('%s params'%str(k))
		
		#SNIP
		struct_mask = circuit_SNIP(net, dataloader, num_params_to_keep=k, feature_targets = feature_targets, feature_targets_coefficients = feature_targets_coefficients, structure=structure, mask=mask, full_dataset = full_dataset, device=device,setup_net=False)
		if structure is not 'weights':
			mask = expand_structured_mask(struct_mask,net) #this weight mask will get applied to the network on the next iteration
		else:
			mask = struct_mask
	apply_mask(net,mask)

	return struct_mask






		

def snip_scores(net,dataloader, feature_targets = None, feature_targets_coefficients = None, full_dataset = True, device=None, structure='weights', criterion= None, setup_net=False):
	###Net should be a preset up, masked model

	
	assert structure in ('weights','kernels','filters')    
	
	if device is None:
		device = 'cuda' if torch.cuda.is_available() else 'cpu'
	
	
	##Calculate Scores
	
	#do we iterate through the whole dataset or not
	iter_dataloader = iter(dataloader)
	
	iters = 1
	if full_dataset:
		iters = len(iter_dataloader)
	
	
	grads = [] #computed scores
	
	for it in range(iters):
		clear_feature_targets_from_net(net)
		
		# Grab a single batch from the training dataset
		inputs, targets = next(iter_dataloader)
		inputs = inputs.to(device)
		targets = targets.to(device)




		# Compute gradients (but don't apply them)
		net.zero_grad()
		
		#Run model forward until all targets reached
		try:
			outputs = net.forward(inputs)
		except TargetReached:
			pass
		
		#get proper loss
		if feature_targets is None:
			loss = criterion(outputs, targets)
		else:   #the real target is feature values in the network
			loss = get_feature_target_from_net(net, feature_targets, feature_targets_coefficients,device=device)
		
		loss.backward()

		#get weight-wise scores
		if grads == []:
			for layer in net.modules():
				if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
					grads.append(torch.abs(layer.weight_mask.grad))
					if layer.last_layer:
						break
		else:
			count = 0
			for layer in net.modules():
				if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
					grads[count] += torch.abs(layer.weight_mask.grad)        
					count += 1
					if layer.last_layer:
						break
				
				
	#structure scoring by weights, kernels, or filters   
	
	if structure == 'weights':
		structure_grads = grads
	elif structure == 'kernels':
		structure_grads = []
		for grad in grads:
			if len(grad.shape) == 4: #conv2d layer
				structure_grads.append(torch.mean(grad,dim = (2,3))) #average across height and width of each kernel
	else:
		structure_grads = []
		for grad in grads:
			if len(grad.shape) == 4: #conv2d layer
				structure_grads.append(torch.mean(grad,dim = (1,2,3))) #average across channel height and width of each filter
				
		
	return structure_grads


def mask_from_sparsity(rank_list, k, random_mask=False):

	if random_mask:
		random_rank_list = []
		for g in rank_list:
			random_rank_list.append(torch.rand(g.shape))
		rank_list = random_rank_list

	all_scores = torch.cat([torch.flatten(x) for x in rank_list])
	norm_factor = torch.sum(abs(all_scores))
	all_scores.div_(norm_factor)

	all_scores = all_scores.type(torch.float)

	threshold, _ = torch.topk(all_scores, k, sorted=True)
	acceptable_score = threshold[-1]
	cum_sal = torch.sum(threshold)

	if acceptable_score == 0:
		print('gradients from this feature are sparse,\
the minimum acceptable rank at this sparsity has a score of zero! \
we will return a mask thats smaller than you asked, by masking all \
parameters with a score of zero.')


	mask = []

	for g in rank_list:
		mask.append(((g / norm_factor) > acceptable_score).float())
		
	return mask,cum_sal




def kernel_mask_2_effective_kernel_mask(kernel_mask):
	effective_mask = deepcopy(kernel_mask)
	for i in range(len(effective_mask)):
		effective_mask[i] = effective_mask[i].to('cpu')

	#pixel to feature connectivity check
	prev_filter_mask = None
	for i in range(len(effective_mask)):
		if prev_filter_mask is not None:  # if we arent in first layer, we have to eliminate kernels connecting to 'dead' filters from the previous layer
			effective_mask[i] = prev_filter_mask*effective_mask[i]

		#now we need to get the filter mask for this layer, (masking those filters with no kernels leading in)
		prev_filter_mask = torch.zeros(effective_mask[i].shape[0])
		for j in range(effective_mask[i].shape[0]):
			if not torch.all(torch.eq(effective_mask[i][j],torch.zeros(effective_mask[i].shape[1]))):
				prev_filter_mask[j] = 1

	#reverse direction feature to pixel connectivity check
	prev_filter_mask = None
	for i in reversed(range(len(effective_mask))):
		if prev_filter_mask is not None:  # if we arent in last layer, we have to eliminate kernels connecting to 'dead' filters from the previous layer
			effective_mask[i] = torch.transpose(prev_filter_mask*torch.transpose(effective_mask[i],0,1),0,1)

		#now we need to get the filter mask for this layer, (masking those filters with no kernels leading in)
		prev_filter_mask = torch.zeros(effective_mask[i].shape[1])
		for j in range(effective_mask[i].shape[1]):
			if not torch.all(torch.eq(effective_mask[i][:,j],torch.zeros(effective_mask[i].shape[0]))):
				prev_filter_mask[j] = 1

	orig_sum  = 0
	for l in kernel_mask:
		orig_sum += int(torch.sum(l))
	print('original mask: %s kernels'%str(orig_sum))

	effective_sum  = 0
	for l in effective_mask:
		effective_sum += int(torch.sum(l))
	print('effective mask: %s kernels'%str(effective_sum))

	return effective_mask


def structured_mask_from_mask(mask, structure = 'kernels'):
	
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

	return out_mask



def circuit_kernel_magnitude_ranking(model, feature_targets = None,feature_targets_coefficients = None,random_ranks = False, device=None, structure='kernels', mask=None, setup_net=True): 

	
	assert structure in ('weights','kernels','filters')
	
	if device is None:
		device = 'cuda' if torch.cuda.is_available() else 'cpu'

	model.to('cpu').eval()
	net = deepcopy(model)
	net = net.to(device).eval()	
	for param in net.parameters():
		param.requires_grad = False
	
	if setup_net:
		setup_net_for_circuit_prune(net, feature_targets=feature_targets)


	rank_list = []

	for layer in net.modules():
		if isinstance(layer, nn.Conv2d):
			if not layer.last_layer:
				w = layer.weight.cpu()
				w_abs = torch.abs(w)
				rank_list.append(w_abs.mean(dim=(2, 3)))
			else:
				w_empty = torch.zeros(layer.weight.shape[0],layer.weight.shape[1]).cpu()
				w = layer.weight[feature_targets[layer.ref_name][0]].cpu()
				w_abs = torch.abs(w)
				r = w_abs.mean(dim=(1, 2))
				w_empty[feature_targets[layer.ref_name][0]] = r
				rank_list.append(w_empty)
				break

	return rank_list

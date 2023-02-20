import torch.nn as nn
import torch.nn.functional as F
import torch

class dissected_Conv2d(torch.nn.Module):       #2d conv Module class that has presum activation maps as intermediate output

	def __init__(self, from_conv, absolute = True):      # from conv is normal nn.Conv2d object to pull weights and bias from
		super(dissected_Conv2d, self).__init__()
		#self.from_conv = from_conv
		self.in_channels = from_conv.weight.shape[1]
		self.out_channels = from_conv.weight.shape[0]
		self.weight_perm,self.add_perm,self.add_indices = self.gen_inout_permutation()
		self.preadd_conv = self.make_preadd_conv(from_conv)
		self.bias = None
		if from_conv.bias is not None:
			self.bias = nn.Parameter(from_conv.bias.unsqueeze(1).unsqueeze(1))

		self.kernel_scores = None
		self.preadd_out_hook = None
		self.absolute = absolute

	def gen_inout_permutation(self):
		'''
		When we flatten out all the output channels not to be grouped by 'output channel', we still want the outputs sorted
		such that they can be conveniently added based on input channel later
		'''
		in_chan = self.in_channels
		out_chan = self.out_channels
		
		weight_perm = []
		for i in range(in_chan):
			for j in range(out_chan):
				weight_perm.append(i+j*in_chan)
		
		add_perm = []
		add_indices = {}
		for o in range(out_chan):
			add_indices[o] = []
			for i in range(in_chan):
				add_perm.append(o+i*out_chan)
				add_indices[o].append(o+i*out_chan)
		return torch.LongTensor(weight_perm),torch.LongTensor(add_perm),add_indices


	def make_preadd_conv(self,from_conv):
		'''
		nn.Conv2d takes in 'in_channel' number of feature maps, and outputs 'out_channel' number of maps. 
		internally it has in_channel*out_channel number of 2d conv kernels. Normally, featuremaps associated 
		with a particular output channel resultant from these kernel convolution are all added together,
		this function changes a nn.Conv2d module into a module where this final addition doesnt happen. 
		The final addition can be performed seperately with permute_add_feature_maps.
		'''
		in_chan = self.in_channels
		out_chan = self.out_channels
		
		kernel_size = from_conv.kernel_size
		padding = from_conv.padding
		stride = from_conv.stride
		new_conv = nn.Conv2d(in_chan,in_chan*out_chan,kernel_size = kernel_size,
							 bias = False, padding=padding,stride=stride,groups= in_chan)
		new_conv.weight = torch.nn.parameter.Parameter(
				from_conv.weight.view(in_chan*out_chan,1,kernel_size[0],kernel_size[1])[self.weight_perm])
		return new_conv

		
	def permute_add_featuremaps(self,feature_map):
		'''
		Perform the sum within output channels step.  (THIS NEEDS TO BE SPEED OPTIMIZED)
		'''
		x = feature_map
		x = x[:, self.add_perm, :, :]
		x = torch.split(x.unsqueeze(dim=1),self.in_channels,dim = 2)
		x = torch.cat(x,dim = 1)
		x = torch.sum(x,dim=2)
		return x
	

	def compute_kernel_scores(self,grad):
		activation = self.preadd_out
		if self.absolute:
			self.kernel_scores = torch.abs(activation * grad).mean(dim=(2,3))
		else:
			self.kernel_scores = (activation * grad).mean(dim=(2,3))
		self.kernel_scores = self.kernel_scores.sum(dim=0).detach().cpu()
		self.preadd_out = None #clean up, memory is precious
		
	def unflatten_kernel_scores(self, scores = None):
		'''
		kernel scores have been flattened, which kernel corresponds with 
		which in and out channels? Here we unflatten
		so scores are of shape out_channel x in_channel again
		'''

		if scores is None:
			scores = self.kernel_scores

		out_list = []
		for out_chan in self.add_indices:
			in_list = []
			for in_chan in self.add_indices[out_chan]:
				in_list.append(scores[in_chan].unsqueeze(dim=0).unsqueeze(dim=0))                 
			out_list.append(torch.cat(in_list,dim=1))
		unflattened_scores = torch.cat(out_list,dim=0)
		return unflattened_scores

						
	def forward(self, x):

		if self.kernel_scores is not None: self.kernel_scores = None

		preadd_out = self.preadd_conv(x)  #get output of convolutions

		self.preadd_out = preadd_out
 
		#Set hooks for calculating rank on backward pass
		#if self.store_ranks:
		#	self.preadd_out = preadd_out
		if self.preadd_out_hook is not None:
			self.preadd_out_hook.remove()
		self.preadd_out_hook = self.preadd_out.register_hook(self.compute_kernel_scores)
			#if self.preadd_ranks is not None:
			#    print(self.preadd_ranks.shape)

		added_out = self.permute_add_featuremaps(preadd_out)    #add convolution outputs by output channel
		if self.bias is not None:  
			postbias_out = added_out + self.bias
		else:
			postbias_out = added_out

		return postbias_out




# takes a full model and replaces all conv2d instances with dissected conv 2d instances
def dissect_model(model,mod_names = [],absolute=True):

	for name, module in reversed(model._modules.items()):
		if len(list(module.children())) > 0:
			mod_names.append(str(name))
			# recurse
			model._modules[name] = dissect_model(module,mod_names =mod_names, absolute=absolute)
			mod_names.pop()

		if isinstance(module, torch.nn.modules.conv.Conv2d):    # found a 2d conv module to transform
			new_module = dissected_Conv2d(module) 
			model._modules[name] = new_module

		elif isinstance(module, torch.nn.modules.Dropout):    #make dropout layers not dropout  #also set batchnorm to eval
			model._modules[name].eval() 


		else:    #make activation functions not 'inplace'
			model._modules[name].inplace=False                    

	return model









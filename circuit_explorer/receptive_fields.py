import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F


def check_same(stride):
	if isinstance(stride, (list, tuple)):
		assert len(stride) == 2 and stride[0] == stride[1]
		stride = stride[0]
	return stride

def receptive_field(model, input_size, batch_size=-1, device="cuda",print_output=True):
	'''
	:parameter
	'input_size': tuple of (Channel, Height, Width)
	:return  OrderedDict of `Layername`->OrderedDict of receptive field stats {'j':,'r':,'start':,'conv_stage':,'output_shape':,}
	'j' for "jump" denotes how many pixels do the receptive fields of spatially neighboring units in the feature tensor
		do not overlap in one direction.
		i.e. shift one unit in this feature map == how many pixels shift in the input image in one direction.
	'r' for "receptive_field" is the spatial range of the receptive field in one direction.
	'start' denotes the center of the receptive field for the first unit (start) in on direction of the feature tensor.
		Convention is to use half a pixel as the center for a range. center for `slice(0,5)` is 2.5.
	'''

	#added by Chris
	try:
		model = model.features
	except:
		pass

	def register_hook(module):

		def hook(module, input, output):
			class_name = str(module.__class__).split(".")[-1].split("'")[0]
			module_idx = len(receptive_field)
			m_key = "%i" % module_idx
			p_key = "%i" % (module_idx - 1)
			receptive_field[m_key] = OrderedDict()

			if not receptive_field["0"]["conv_stage"]:
				print("Enter in deconv_stage")
				receptive_field[m_key]["j"] = 0
				receptive_field[m_key]["r"] = 0
				receptive_field[m_key]["start"] = 0
			else:
				p_j = receptive_field[p_key]["j"]
				p_r = receptive_field[p_key]["r"]
				p_start = receptive_field[p_key]["start"]
				
				if class_name == "Conv2d" or class_name == "MaxPool2d":
					kernel_size = module.kernel_size
					stride = module.stride
					padding = module.padding
					dilation = module.dilation
	   
					kernel_size, stride, padding, dilation = map(check_same, [kernel_size, stride, padding, dilation])
					receptive_field[m_key]["j"] = p_j * stride
					receptive_field[m_key]["r"] = p_r + ((kernel_size - 1) * dilation) * p_j
					receptive_field[m_key]["start"] = p_start + ((kernel_size - 1) / 2 - padding) * p_j
				elif class_name == "BatchNorm2d" or class_name == "ReLU" or class_name == "Bottleneck":
					receptive_field[m_key]["j"] = p_j
					receptive_field[m_key]["r"] = p_r
					receptive_field[m_key]["start"] = p_start
				elif class_name == "ConvTranspose2d":
					receptive_field["0"]["conv_stage"] = False
					receptive_field[m_key]["j"] = 0
					receptive_field[m_key]["r"] = 0
					receptive_field[m_key]["start"] = 0
				else:
					raise ValueError("module not ok")
					pass
			receptive_field[m_key]["input_shape"] = list(input[0].size()) # only one
			receptive_field[m_key]["input_shape"][0] = batch_size
			if isinstance(output, (list, tuple)):
				# list/tuple
				receptive_field[m_key]["output_shape"] = [
					[-1] + list(o.size())[1:] for o in output
				]
			else:
				# tensor
				receptive_field[m_key]["output_shape"] = list(output.size())
				receptive_field[m_key]["output_shape"][0] = batch_size

		if (
			not isinstance(module, nn.Sequential)
			and not isinstance(module, nn.ModuleList)
			and not (module == model)
		):
			hooks.append(module.register_forward_hook(hook))

	device = device.lower()
	assert device in [
		"cuda",
		"cpu",
	], "Input device is not valid, please specify 'cuda' or 'cpu'"

	if device == "cuda" and torch.cuda.is_available():
		dtype = torch.cuda.FloatTensor
	else:
		dtype = torch.FloatTensor

	# check if there are multiple inputs to the network
	if isinstance(input_size[0], (list, tuple)):
		x = [Variable(torch.rand(2, *in_size)).type(dtype) for in_size in input_size]
	else:
		x = Variable(torch.rand(2, *input_size)).type(dtype)

	# create properties
	receptive_field = OrderedDict()
	receptive_field["0"] = OrderedDict()
	receptive_field["0"]["j"] = 1.0
	receptive_field["0"]["r"] = 1.0
	receptive_field["0"]["start"] = 0.5
	receptive_field["0"]["conv_stage"] = True
	receptive_field["0"]["output_shape"] = list(x.size())
	receptive_field["0"]["output_shape"][0] = batch_size
	hooks = []

	# register hook
	model.apply(register_hook)

	# make a forward pass
	model(x)

	# remove these hooks
	for h in hooks:
		h.remove()

	if print_output:
		print("------------------------------------------------------------------------------")
		line_new = "{:>20}  {:>10} {:>10} {:>10} {:>15} ".format("Layer (type)", "map size", "start", "jump", "receptive_field")
		print(line_new)
		print("==============================================================================")
		total_params = 0
		total_output = 0
		trainable_params = 0
		for layer in receptive_field:
			# input_shape, output_shape, trainable, nb_params
			assert "start" in receptive_field[layer], layer
			assert len(receptive_field[layer]["output_shape"]) == 4
			line_new = "{:7} {:12}  {:>10} {:>10} {:>10} {:>15} ".format(
				"",
				layer,
				str(receptive_field[layer]["output_shape"][2:]),
				str(receptive_field[layer]["start"]),
				str(receptive_field[layer]["j"]),
				format(str(receptive_field[layer]["r"]))
			)
			print(line_new)

		print("==============================================================================")
	# add input_shape
	receptive_field["input_size"] = input_size
	return receptive_field


def receptive_field_for_unit(receptive_field_dict, layer, unit_position):
	"""Utility function to calculate the receptive field for a specific unit in a layer
		using the dictionary calculated above
	:parameter
		'layer': layer name, should be a key in the result dictionary
		'unit_position': spatial coordinate of the unit (H, W)
	```
	alexnet = models.alexnet()
	model = alexnet.features.to('cuda')
	receptive_field_dict = receptive_field(model, (3, 224, 224))
	receptive_field_for_unit(receptive_field_dict, "8", (6,6))
	```
	Out: [(62.0, 161.0), (62.0, 161.0)]
	"""
	if ('feature' not in list(receptive_field_dict.keys())[0]) and ('feature' in layer):
		layer = str(int(layer.split('_')[-1].split('.')[-1])+1) #hack
	input_shape = receptive_field_dict["input_size"]
	if layer in receptive_field_dict:
		rf_stats = receptive_field_dict[layer]
		assert len(unit_position) == 2
		feat_map_lim = rf_stats['output_shape'][2:]
		if np.any([unit_position[idx] < 0 or
				   unit_position[idx] >= feat_map_lim[idx]
				   for idx in range(2)]):
			raise Exception("Unit position outside spatial extent of the feature tensor ((H, W) = (%d, %d)) " % tuple(feat_map_lim))
		# X, Y = tuple(unit_position)
		rf_range = [(rf_stats['start'] + idx * rf_stats['j'] - rf_stats['r'] / 2,
			rf_stats['start'] + idx * rf_stats['j'] + rf_stats['r'] / 2) for idx in unit_position]
		if len(input_shape) == 2:
			limit = input_shape
		else:  # input shape is (channel, H, W)
			limit = input_shape[1:3]
		rf_range = [(max(0, rf_range[axis][0]), min(limit[axis], rf_range[axis][1])) for axis in range(2)]
		#print("Receptive field size for layer %s, unit_position %s,  is \n %s" % (layer, unit_position, rf_range))
		return rf_range
	else:
		raise KeyError("Layer name incorrect, or not included in the model.")


totensor = transforms.ToTensor()
topil = transforms.ToPILImage()

class receptive_field_fit_transform:
	def __init__(self, layer, target_position,recep_field_params=None,model=None,image_size=(3,244,244),device='cpu',shrinkage=1):
		'''
		shrinkage: 0<i<=1, the factor by which the image is shrunk inside the receptive field ("1" fits the image perfectly inside)
		'''
		if (recep_field_params is None) and (model is None):
			raise ValueError('both argument "recep_field_params" and "model" are none, must specify at least one of these.')
		if recep_field_params is None:
			recep_field_params = receptive_field(model, image_size, device=device)
	 
		self.recep_field_params = recep_field_params

		self.image_size = image_size
		self.target_position = target_position
		self.layer = layer
		self.shrinkage = shrinkage

		self.recep_field = receptive_field_for_unit(recep_field_params, layer, target_position)
		print(self.recep_field)
		#self.recep_resize = transforms.Resize((int(self.recep_field[0][1]-self.recep_field[0][0]),int(self.recep_field[1][1]-self.recep_field[1][0])))

	def __call__(self, img_tensor):
		#blank backgrounds
		out_tensor = torch.zeros(3,224,224)
		
		#shrunk images to size of receptive field
		height_window = self.recep_field[0][1]-self.recep_field[0][0]
		width_window = self.recep_field[1][1]-self.recep_field[1][0]
		shrunk_img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(int(height_window*self.shrinkage),int(width_window*self.shrinkage)),mode='bilinear').squeeze(0)
		#shrunk_img_tensor = self.recep_resize(img_tensor)

		#position shrunk images on blanks at receptive field position
		if self.shrinkage == 1:
			out_tensor[:,int(self.recep_field[0][0]):int(self.recep_field[0][1]),int(self.recep_field[1][0]):int(self.recep_field[1][1])] = shrunk_img_tensor
		else:
			height = shrunk_img_tensor.shape[1]
			width = shrunk_img_tensor.shape[2]
			height_start = int(self.recep_field[0][0]+(height_window-height)/2)
			width_start = int(self.recep_field[1][0]+(width_window-width)/2)
			out_tensor[:,height_start:height_start+height,width_start:width_start+width] = shrunk_img_tensor


		return out_tensor
		# #check resultant images (whats actually going to get fed to the model)
		# print(size)
		# recep_circle_img = topil(circle_tensor)
		# recep_circle_wt_border = ImageOps.expand(recep_circle_img, border=border, fill='black')
		# display(recep_circle_wt_border)

		# # normalize and unsqueeze for passing through model
		# circle_tensor = norm(circle_tensor)
		# circle_tensor = torch.unsqueeze(circle_tensor,0)
		# circle_tensor = circle_tensor.to(device)


def recep_field_crop(image,model,layer,target_position,rf_dict = None):
	"""
	inputs: a tensor image, model, layer name, and position in the layers activation map (H,W)
	outputs: cropped image at receptive field for that image
	"""
	if rf_dict is None:
		input_size = tuple(image.shape)
		rf_dict = receptive_field(model, input_size)
	
	pos = receptive_field_for_unit(rf_dict, layer, target_position)
	return image[:,int(pos[0][0]):int(pos[0][1]),int(pos[1][0]):int(pos[1][1])]

def position_crop_image(image,position,layer_name,model=None,input_size=(3,224,244),rf_dict=None):
	'''
	crops a PIL image at the receptive field for an individual unit
	'''

	if rf_dict is None:
		#dont use nested model, for example in Alexnet pass model.features as 'model'
		rf_dict = receptive_field(model, input_size,print_output=False)
	load_image =   transforms.Compose([
									transforms.Resize((input_size[1],input_size[2])),
									transforms.ToTensor()])
	topil = transforms.ToPILImage()

	tensor_image = load_image(image)

	cropped_tensor_image = recep_field_crop(tensor_image,model,layer_name,position,rf_dict = rf_dict) #function from circuit_explorer.receptive_fields
	img = topil(cropped_tensor_image)

	return img

	


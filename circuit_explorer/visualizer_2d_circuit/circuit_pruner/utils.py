#MISC UTILITY FUNCTIONS
import sys
import os
from PIL import Image
import torch
from torch import nn
from circuit_pruner.dissected_Conv2d import *
from copy import deepcopy


### IMAGE PROCESSING ###

def get_image_path(image_name,params):
	found = False
	path = None
	if image_name in params['input_image_list']:
		found = True
		path = params['input_image_directory']+'/'+image_name
	elif image_name in os.listdir(params['prepped_model_path']+'/visualizations/images/'):
		found = True
		path = params['prepped_model_path']+'/visualizations/images/'+image_name
	return found, path


def preprocess_image(image_path,params):
	preprocess = params['preprocess']

	#image loading 
	image_name = image_path.split('/')[-1]
	image = Image.open(image_path)
	image = preprocess(image).float()
	image = image.unsqueeze(0)
	image = image.to(params['device'])
	return image


def jpg_convert_image_folder(path):
	from PIL import Image
	import os
	images = os.listdir(path)

	for image in images:
	
		im = Image.open(os.path.join(path,image))
		im = im.convert("RGB")

		im_name_split = image.split('.')
		im_name_root = '.'.join(im_name_split[:-1])

		im.save(os.path.join(path,im_name_root+'.jpg'))




#### NAMING  ####

#return list of names for conv modules based on their nested module names '_' seperated
def get_conv_full_names(model,mod_names = [],mod_full_names = []):
	#gen names based on nested modules
	for name, module in model._modules.items():
		if len(list(module.children())) > 0:
			mod_names.append(str(name))
			# recurse
			mod_full_names = get_conv_full_names(module,mod_names = mod_names, mod_full_names = mod_full_names)
			mod_names.pop()

		if isinstance(module, torch.nn.modules.conv.Conv2d):    # found a 2d conv module
			mod_full_names.append('_'.join(mod_names+[name]))
			#new_module = dissected_Conv2d(module, name='_'.join(mod_names+[name]), store_activations=store_activations,store_ranks=store_ranks,clear_ranks=clear_ranks,cuda=cuda,device=device) 
			#model._modules[name] = new_module
	return mod_full_names     


#get weights from model as list
def get_weight_list(model,weights = []):
	#gen names based on nested modules
	for name, module in model._modules.items():
		if len(list(module.children())) > 0:
			# recurse
			weights =  get_weight_list(module,weights=weights)

		if isinstance(module, torch.nn.modules.conv.Conv2d) or isinstance(module, torch.nn.Linear):    # found a 2d conv module
			weights.append(module.weight.detach().cpu())
			#new_module = dissected_Conv2d(module, name='_'.join(mod_names+[name]), store_activations=store_activations,store_ranks=store_ranks,clear_ranks=clear_ranks,cuda=cuda,device=device) 
			#model._modules[name] = new_module
	return weights   


def ref_name_modules(net):
	
	# recursive function to get layers
	def name_layers(net, prefix=[]):
		if hasattr(net, "_modules"):
			for name, layer in net._modules.items():

				if layer is None:
					# e.g. GoogLeNet's aux1 and aux2 layers
					continue
				
				layer.ref_name = "_".join(prefix + [name])
				
				name_layers(layer,prefix=prefix+[name])

	name_layers(net)


def show_model_layer_names(model, getLayerRepr=False,printer=True):
	"""
	If getLayerRepr is True, return a OrderedDict of layer names, layer representation string pair.
	If it's False, just return a list of layer names
	"""
	
	layers = OrderedDict() if getLayerRepr else []
	conv_linear_layers = []
	# recursive function to get layers
	def get_layers(net, prefix=[]):
		
		if hasattr(net, "_modules"):
			for name, layer in net._modules.items():

				if layer is None:
					# e.g. GoogLeNet's aux1 and aux2 layers
					continue
				if getLayerRepr:
					layers["_".join(prefix+[name])] = layer.__repr__()
				else:
					layers.append("_".join(prefix + [name]))
				
				if isinstance(layer, nn.Conv2d):
					conv_linear_layers.append(("_".join(prefix + [name]),'  conv'))
				elif isinstance(layer, nn.Linear):
					conv_linear_layers.append(("_".join(prefix + [name]),'  linear'))
					
				get_layers(layer, prefix=prefix+[name])
				
	get_layers(model)
	
	if printer:
		print('All Layers:\n')
		for layer in layers:
			print(layer)

		print('\nConvolutional and Linear layers:\n')
		for layer in conv_linear_layers:
			print(layer)

	return layers


def get_model_conv_weights(model):
	weights = []
	# recursive function to get layers
	def get_weights(module):
		if hasattr(module, "_modules"):
			for name, layer in module._modules.items():

				if layer is None:
					# e.g. GoogLeNet's aux1 and aux2 layers
					continue
				
				if isinstance(layer, nn.Conv2d):
					weights.append(layer.weight.detach().cpu())

				get_weights(layer)

	get_weights(model)

	return weights


def get_model_filterids(model):
	ref_name_modules(model)
	
	out = []
	
	next_filterid = 0
	def get_ids(module, next_filterid = 0):

		if hasattr(module, "_modules"):
			for name, layer in module._modules.items():

				if layer is None:
					# e.g. GoogLeNet's aux1 and aux2 layers
					continue
				if isinstance(layer, nn.Conv2d):
					num_filters = layer.weight.shape[0]
					out.append([layer.ref_name,list(range(next_filterid,next_filterid+num_filters))])
					next_filterid = next_filterid+num_filters

				get_ids(layer, next_filterid = next_filterid)

	get_ids(model)
	return out


def filterid_2_perlayerid(filterid,model,imgnode_names = ['r','b','g']):    #takes in node unique id outputs tuple of layer and within layer id
	layer_nodes = get_model_filterids(model)
	if isinstance(filterid,str):
		if not filterid.isnumeric():
			layer = 'img'
			layer_name='img'
			within_layer_id = imgnode_names.index(filterid)
			return layer,within_layer_id, layer_name
	filterid = int(filterid)
	total= 0
	for i in range(len(layer_nodes)):
		total += len(layer_nodes[i][1])
		if total > filterid:
			layer = i
			layer_name = layer_nodes[i][0]
			within_layer_id = layer_nodes[i][1].index(filterid)
			break
	#layer = nodes_df[nodes_df['category']=='overall'][nodes_df['node_num'] == nodeid]['layer'].item()
	#within_layer_id = nodes_df[nodes_df['category']=='overall'][nodes_df['node_num'] == nodeid]['node_num_by_layer'].item()
	return layer,within_layer_id,layer_name

	
#return list of names for conv modules based on their simple order, first conv is 'conv1', then 'conv2' etc. 
def get_conv_simple_names(model):
	names = []
	count = 0
	for layer in model.modules():
		if isinstance(layer, nn.Conv2d):
			names.append('conv'+str(count))
			count+=1
	return names
 
# returns a dict that maps simple names to full names
def gen_conv_name_dict(model):
	simple_names = get_conv_simple_names(model)
	full_names = get_conv_full_names(model)
	return dict(zip(simple_names, full_names))


def nodeid_2_perlayerid(nodeid,params):    #takes in node unique id outputs tuple of layer and within layer id
	imgnode_names = params['imgnode_names']
	layer_nodes = params['layer_nodes']
	if isinstance(nodeid,str):
		if not nodeid.isnumeric():
			layer = 'img'
			layer_name='img'
			within_layer_id = imgnode_names.index(nodeid)
			return layer,within_layer_id, layer_name
	nodeid = int(nodeid)
	total= 0
	for i in range(len(layer_nodes)):
		total += len(layer_nodes[i][1])
		if total > nodeid:
			layer = i
			layer_name = layer_nodes[i][0]
			within_layer_id = layer_nodes[i][1].index(nodeid)
			break
	#layer = nodes_df[nodes_df['category']=='overall'][nodes_df['node_num'] == nodeid]['layer'].item()
	#within_layer_id = nodes_df[nodes_df['category']=='overall'][nodes_df['node_num'] == nodeid]['node_num_by_layer'].item()
	return layer,within_layer_id,layer_name

def layernum2name(layer,offset=1,title = 'layer'):
	return title+' '+str(layer+offset)


def check_edge_validity(nodestring,params):
	from_node = nodestring.split('-')[0]
	to_node = nodestring.split('-')[1]
	try:
		from_layer,from_within_id,from_layer_name = nodeid_2_perlayerid(from_node,params)
		to_layer,to_within_id,to_layer_name = nodeid_2_perlayerid(to_node,params)
		#check for valid edge
		valid_edge = False
		if from_layer=='img':
			if to_layer== 0:
				valid_edge = True
		elif to_layer == from_layer+1:
			valid_edge = True
		if not valid_edge:
			print('invalid edge name')
			return [False, None, None, None, None]
		return True, from_layer,to_layer,from_within_id,to_within_id
	except:
		#print('exception')
		return [False, None, None, None, None] 

def edgename_2_singlenum(model,edgename,params):
	valid, from_layer,to_layer,from_within_id,to_within_id = check_edge_validity(edgename,params)
	if not valid:
		raise ValueError('edgename %s is invalid'%edgename)
	conv_module = layer_2_dissected_conv2d(int(to_layer),model)[0]
	return conv_module.add_indices[int(to_within_id)][int(from_within_id)]


### TENSORS ###

def unravel_index(indices,shape):
	r"""Converts flat indices into unraveled coordinates in a target shape.

	This is a `torch` implementation of `numpy.unravel_index`.

	Args:
		indices: A tensor of (flat) indices, (*, N).
		shape: The targeted shape, (D,).

	Returns:
		The unraveled coordinates, (*, N, D).
	"""

	coord = []

	for dim in reversed(shape):
		coord.append(indices % dim)
		indices = indices // dim

	coord = torch.stack(coord[::-1], dim=-1)

	return coord


###  NETWORKS ###

def relu(array):
	neg_indices = array < 0
	array[neg_indices] = 0
	return array


### COLOR

def rgb2hex(r, g, b):
	return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def color_vec_2_str(colorvec,a='1'):
	return 'rgba(%s,%s,%s,%s)'%(str(int(colorvec[0])),str(int(colorvec[1])),str(int(colorvec[2])),a)


### PATH ###

def update_sys_path(path):
	full_path = os.path.abspath(path)
	if full_path not in sys.path:
		sys.path.insert(0,full_path)

def load_config(config_path):
	if '/' in config_path:
		config_root_path = ('/').join(config_path.split('/')[:-1])
		update_sys_path(config_root_path)
	config_module = config_path.split('/')[-1].replace('.py','')
	config = __import__(config_module)
	return config



### TRULY MISC ###

def get_nth_element_from_nested_list(l,n):    #this seems to come up with the nested layer lists
	flat_list = [item for sublist in l for item in sublist]
	return flat_list[n]


def minmax_normalize_between_values(vec,min_v,max_v):
	return (max_v-min_v)*(vec-np.min(vec))/(np.max(vec)-np.min(vec))+min_v
	
def min_distance(x,y,minimum=1):
	dist = np.linalg.norm(x-y)
	if dist > minimum:
		return dist,True
	else:
		return dist,False
	
def multipoint_min_distance(points):   #takes numpy array of shape (# points, # dimensions)
	dist_mat = distance_matrix(points,points)
	dist_mat[np.tril_indices(dist_mat.shape[0], 0)] = 10000
	print(dist_mat)
	return np.min(dist_mat)


def mask_intersect_over_union(mask1,mask2):
	iou = []
	for i in range(len(mask1)):
		intersect_mask = mask1[i]*mask2[i]
		union_mask = torch.ceil((mask1[i]+mask2[i])/2)
		iou.append(torch.sum(intersect_mask)/torch.sum(union_mask))
	return iou
	

def plot_iou_from_masks(mask1,mask2,big=True):
	import plotly.express as px
	import pandas as pd
	layer_IoU = mask_intersect_over_union(mask1,mask2)

	Layer = []

	for i in range(len(layer_IoU)):
		Layer.append(str(i+1))

	import plotly.graph_objects as go

	if big:
		m_size = 20
	else:
		m_size= 5

	fig = go.Figure()
	fig.add_trace(go.Scatter(x=Layer, y=layer_IoU, fill='tozeroy',marker=dict(size=m_size),line_color=px.colors.qualitative.T10[0])) # fill down to xaxis
	fig.update_layout({ 'width':500,
						'plot_bgcolor':'rgba(255,255,255,1)',
						'paper_bgcolor':'rgba(255,255,255,1)',
						#'font_size':20
						})
	fig.update_yaxes(range=[0, 1],title_text='IoU')
	fig.update_xaxes(title_text='Layer')
	#fig.show()
	return fig
	
	





def circuit_2_model_sparsity(circuit,model,use_kernel_sparsity=True):
	'''
	sometimes we extract a subcircuit from a circuit, and then the sparsity is with respect to 
	the circuit, not the original model. This function provides a factor of the circuit size to 
	the original model size, so just multiply the subcircuit sparsity by the factor this 
	function returns to get the sparsity of the subcircuit with respect to the orig model.

	'''
	from collections import OrderedDict

	ref_name_modules(circuit)
	ref_name_modules(model)
	circuit_conv_dims = OrderedDict()
	model_conv_dims = OrderedDict()

	# recursive function to get layers
	def get_dims(net,conv_dims,use_kernel_sparsity=use_kernel_sparsity,total_params=0,num_zero_params=0):
		if hasattr(net, "_modules"):
			for name, layer in net._modules.items():

				if layer is None:
					# e.g. GoogLeNet's aux1 and aux2 layers
					continue
				
				if isinstance(layer, nn.Conv2d):
					conv_dims[layer.ref_name] = [layer.weight.shape[0],layer.weight.shape[1]]
					if use_kernel_sparsity:
						total_params += int(layer.weight.shape[0]*layer.weight.shape[1])
						kernel_sums = torch.sum(torch.abs(layer.weight), (2,3), keepdim=False)
						num_zero_params += int((kernel_sums == 0).sum())					
				if use_kernel_sparsity:
					kernel_sparsity, conv_dims = get_dims(layer,conv_dims,use_kernel_sparsity=use_kernel_sparsity,total_params=total_params,num_zero_params=num_zero_params)
				else:
					conv_dims = get_dims(layer,conv_dims,use_kernel_sparsity=use_kernel_sparsity,total_params=total_params,num_zero_params=num_zero_params)

		if use_kernel_sparsity:
			kernel_sparsity = (total_params-num_zero_params)/total_params
			return kernel_sparsity, conv_dims
		else:
			return conv_dims

	if use_kernel_sparsity:
		kernel_sparsity, circuit_conv_dims = get_dims(circuit, circuit_conv_dims)
	else:
		circuit_conv_dims = get_dims(circuit, circuit_conv_dims)
	model_conv_dims = get_dims(model, model_conv_dims,use_kernel_sparsity=False)

	circuit_size = 0
	model_size = 0

	last_feat = next(reversed(circuit_conv_dims))
	for feat in circuit_conv_dims:
		circuit_size += circuit_conv_dims[feat][0]*circuit_conv_dims[feat][1]
		if feat == last_feat:
			model_size += circuit_conv_dims[feat][0]*model_conv_dims[feat][1]
		else:
			model_size += model_conv_dims[feat][0]*model_conv_dims[feat][1]

	filter_sparsity = circuit_size/model_size
	print('filter sparsity: %s'%str(filter_sparsity))

	if use_kernel_sparsity:
		print('kernel sparsity: %s'%str(kernel_sparsity))
		sparsity = kernel_sparsity*filter_sparsity
	else:
		sparsity = filter_sparsity

	return sparsity


def display_image_patch_for_activation(image_path,layer_name,position,receptive_fields,simple_name=False,frame = True, save=False,image_size=(3,224,224)):
    '''
    image_path -> full path to image
    layer_name -> name of reference layer for activation map (can be a layer name based on _ convention or simple 'conv1' convention)
    position -> a tuple (w,h) of position in activation map for which image patch is the receptive field
    simple_name -> set to true if using 'conv1' 'conv2' naming convention, False otherwise
    '''
    from circuit_pruner.receptive_fields import receptive_field_for_unit
    #if simple_name:
    #    name_dict = gen_conv_name_dict(model)
    #    layer_name = name_dict[layer_name]
    recep_field = receptive_field_for_unit(receptive_fields, layer_name, position)
    
    image = Image.open(image_path)
    #display(image)
    resize_2_tensor = transforms.Compose([transforms.Resize((image_size[1],image_size[2])),transforms.ToTensor()])
    tensor_image = resize_2_tensor(image)
    rand_tensor = torch.zeros(image_size[0],image_size[1],image_size[2])
    cropped_tensor_image = tensor_image[:,int(recep_field[0][0]):int(recep_field[0][1]),int(recep_field[1][0]):int(recep_field[1][1])]
    rand_tensor[:,int(recep_field[0][0]):int(recep_field[0][1]),int(recep_field[1][0]):int(recep_field[1][1])] = cropped_tensor_image
    if frame:
        cropped_image = transforms.ToPILImage()(rand_tensor).convert("RGB")
    else:    
        cropped_image = transforms.ToPILImage()(cropped_tensor_image).convert("RGB")
    
    if save:
        cropped_image.save(save)
    else:
        display(cropped_image)



####SPATIAL
#rotation for mds plots
from scipy.spatial.distance import cdist

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def rotate_cartesian(vec2d,r):    #rotates 2d cartesian coordinates by some radians 
    x,y = vec2d[0], vec2d[1]
    x_out = np.sqrt(x**2+y**2)*np.cos(np.arctan2(y,x)+r)
    y_out = np.sqrt(x**2+y**2)*np.sin(np.arctan2(y,x)+r)
    return np.array([x_out,y_out])


def align_maps(in_map,anchor_map,angles_tested=64):
	#centroid align to origin
	in_map[0] = in_map[0]-in_map[0].mean()
	in_map[1] = in_map[1]-in_map[1].mean()
	flipped_in_map = deepcopy(in_map)
	flipped_in_map[0] = - flipped_in_map[0]
	anchor_map[0] = anchor_map[0]-anchor_map[0].mean()
	anchor_map[1] = anchor_map[1]-anchor_map[1].mean()
	
	#rotate align
	flip=False
	min_dist = 1e40
	min_discrete_angle = 0
	for p in range(angles_tested):
		#unflipped
		test_map = rotate_cartesian(in_map,p*2*np.pi/angles_tested)
		dist = sum(np.diagonal(cdist(np.swapaxes(test_map,0,1),np.swapaxes(anchor_map,0,1))))
		if dist < min_dist:
			flip=False
			min_discrete_angle = p
			min_dist = dist
		#flipped
		test_map = rotate_cartesian(flipped_in_map,p*2*np.pi/angles_tested)
		dist = sum(np.diagonal(cdist(np.swapaxes(test_map,0,1),np.swapaxes(anchor_map,0,1))))
		if dist < min_dist:
			flip=True
			min_discrete_angle = p
			min_dist = dist
	if not flip:	
		return rotate_cartesian(in_map,min_discrete_angle*2*np.pi/angles_tested)
	else:
		return rotate_cartesian(flipped_in_map,min_discrete_angle*2*np.pi/angles_tested)
		

def align_dataframes(in_df,anchor_df,angles_tested=64):
    
    in_map = np.array([list(in_df['x']),list(in_df['y'])])
    anchor_map = np.array([list(anchor_df['x']),list(anchor_df['y'])])
    
    aligned_map = align_maps(in_map,anchor_map,angles_tested=angles_tested)
    in_df['x'],in_df['y'] = aligned_map[0],aligned_map[1]
    
    return in_df
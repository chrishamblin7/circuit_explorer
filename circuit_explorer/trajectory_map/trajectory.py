
from circuit_explorer.data_loading import single_image_data, rank_image_data, default_preprocess
from circuit_explorer.target import sum_abs_loss, positional_loss, layer_activations_from_dataloader
from circuit_explorer.score import actgrad_filter_score, actgrad_filter_extractor, get_num_params_from_cum_score
import umap
import torch
from torch import nn
import numpy as np
import os
from torch.utils.data import DataLoader
import pandas as pd
from collections import OrderedDict
from copy import deepcopy

from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix
import numpy as np
from scipy.spatial.distance import cdist

def umap_from_scores(scores,layers='all',norm_data=False,n_components=2):
  if layers=='all':
    layers = list(scores[0].keys())
  if isinstance(layers,str):
    layers = [layers]

  data = []
  l1_norms = []
  l2_norms = []
  #import pdb;pdb.set_trace()
  for sample in scores:
    traj_v = []
    for l in layers:
      traj_v = traj_v +list(sample[l].flatten())

    l1_norms.append(float(torch.tensor(traj_v).sum()))
    l2_norms.append(float(torch.norm(torch.tensor(traj_v))))

    if norm_data:
      traj_v = list(nn.functional.normalize(torch.tensor(traj_v),dim=0))
    data.append(traj_v) 

  data = np.array(data)
  mapper = umap.UMAP(n_components=n_components).fit(data)
    
  out_data = mapper.fit_transform(data)
  return out_data,l1_norms,l2_norms


def gen_image_trajectory_map_df(data_folder,model,target_layer,unit,
                                scores=None,preprocess=default_preprocess,
                                position=None,norm_data=False,umap_layer='all',
                                batch_size=64, target_layer_activations=None,n_components=2):
  '''
  required arguments:
    data_folder:  path to folder with just images with no sub-folders, oooor (not yet implemented) with class-wise subfolders
    model: A pytorch model
    target_layer: the name for the layer the trajectory map goes to; from "OrderedDict([*model.named_modules()]).keys()"
    unit: either an integer for a basis direction in the target_layer, or a list-like object of floats for a vector direction
  optional arguments:
    scores: scores per image (see simple_api.score), defaults to none nand computing these within the function
    preprocess: a torchvision transform, defaults to 224x224  resize and imagenet normalization
    position: only relevant for convolutional layers, which output an activation map, if position it specifies, it refers to a cell in this map, from which you backprop
    norm_data: do you want to normalize the trajectory vectors before running umap
    umap_layer: excepts a string layer name, like target layer, or a list of layer names. these are the layers whos scores are included in the trajectory vector. Defaults to "all" which includes all layers
  '''

  device = next(model.parameters()).device
  layers = OrderedDict([*model.named_modules()])
  all_images = os.listdir(data_folder)
  all_images.sort()

  if target_layer_activations is None:
    print('getting layer activations')
    target_layer_activations = layer_activations_from_dataloader(target_layer,data_folder,model,batch_size=batch_size)[target_layer]

  if isinstance(unit,int):
    unit_activations = target_layer_activations[:,unit]
  else:
    unit_activations = torch.tensordot(target_layer_activations, torch.tensor(unit).to('cpu').float(), dims=([1],[0]))


  assert not (len(unit_activations.shape)>1 and (position is None))
  # if len(unit_activations.shape>1) and (position is None):
  #   print('you did not specify a position but your layer returns multiple values per feature (it has an activation map). \n \
  #         Well average over this map, but consider specifying a position (as a tuple of ints).')

  if position is not None:
    for i in range(len(position)-1,-1,-1):
      unit_activations = unit_activations[..., position[i]]
  
  data = []
  for i in range(unit_activations.shape[0]):
    data.append({'image':all_images[i],
                'activation':float(unit_activations[i]),
                'layer':target_layer,
                'position':position
                })

  #import pdb; pdb.set_trace()

  if scores is None:
    scores = []
    print('computing imagewise trajectory vectors')
    for i,d in enumerate(data):
        if i%100==0:
            print(str(i)+'/'+str(len(all_images)))
        image_path = os.path.join(data_folder,d['image'])
        dataloader = DataLoader(single_image_data(image_path,
                                                preprocess,
                                                rgb=True),
                                batch_size=1,
                                shuffle=False
                                )
        if position in d.keys():
          position = d['position']
          loss_func = positional_loss(position)
        else:
          loss_func = sum_abs_loss

        #use argument 'score_type == "activations" or score_type == "gradients"' to score with respect to those values per filter instead  
        image_scores = actgrad_filter_score(model,dataloader,target_layer,unit) 
        scores.append(image_scores)
      

    data_map,l1_norms,l2_norms = umap_from_scores(scores,norm_data=norm_data,layers=umap_layer,n_components=n_components) #umap of standarized image-wise trajectories through the network to the target feature
    #make a dataframe of umap data, this will make a consisent format that easier to save as a single object
    #we will load in some of these dataframes from google drive that I generated for each feature later on . . .

    columns = ['x','y','image','activation','l1_norm','l2_norm']
    if n_components == 3:
      columns.append('z')

    big_list = []
    for i in range(len(data)):
        image_name = data[i]['image']
        x = data_map[i][0]
        y = data_map[i][1]
        activation = float(unit_activations[i])
        l1_norm = l1_norms[i]
        l2_norm = l2_norms[i]
        row = [x,y,image_name,activation,l1_norm,l2_norm]
        if n_components == 3:
          row.append(data_map[i][2])
        big_list.append(row)
    print('computing umap projection') 
    umap_df = pd.DataFrame(big_list,columns=columns)

    return umap_df, scores


from circuit_explorer.utils import cart2pol, pol2cart, rotate_cartesian

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


def random_spaced_indices_from_df(df,num):
    '''
    useful for adding images to trajectory map such that they are random and well spaced,
    use as input to image_order argument in "full_app_from_df"
    '''
    pts2D = np.swapaxes(np.array([list(df['x']),list(df['y'])]),0,1)
    kmeans = KMeans(n_clusters=num, random_state=0).fit(pts2D)
    labels = kmeans.predict(pts2D)
    cntr = kmeans.cluster_centers_
    random_images = []
    for i, c in enumerate(cntr):
        lab = np.where(labels == i)[0]
        pts = pts2D[lab]
        d = distance_matrix(c[None, ...], pts)
        idx1 = np.argmin(d, axis=1) + 1
        idx2 = np.searchsorted(np.cumsum(labels == i), idx1)[0]
        random_images.append(idx2)
    return random_images

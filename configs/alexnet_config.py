### Config File for Alexnet Sparse. ###

import torch
import os
import sys
from circuit_explorer import root_path

### NAME ###

name = 'alexnet'


###MODEL###


from torchvision import models
import torch.nn as nn

model = models.alexnet(pretrained=True)


###DATA PATH###

if not os.path.exists(root_path+'/image_data/imagenet_2'):
	from circuit_explorer.download_from_gdrive import download_from_gdrive
	download_from_gdrive('alexnet_sparse',target = 'images')

data_path =  root_path+'/image_data/imagenet_2/'   #Set this to the system path for the folder containing input images you would like to see network activation maps for.

label_file_path = root_path+'/image_data/imagenet_labels.txt'      #line seperated file with names of label classes as they appear in image names
						  #set to None if there are no target classes for your model
						  #make sure the order of labels matches the order in desired target vectors



###DATA PREPROCESSING###

from torchvision import transforms

preprocess =  transforms.Compose([
        						transforms.Resize((224,224)),
        						transforms.ToTensor(),
								transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     	 			 std=[0.229, 0.224, 0.225])])


#targets
layers = ['features.6','features.8','features.10']
units = range(20)

#GPU
device = 'cuda:1'


#AUX 
num_workers = 4     #num workers argument in dataloader
seed = 2            #manual seed
batch_size = 1   #batch size for feeding rank image set through model (input image set is sent through all at once)

### Config File for Alexnet Sparse. ###


import torch
import os
import sys
from circuit_explorer import root_path

### NAME ###

name = 'resnet18'


###MODEL###


from torchvision.models import resnet18
import torch.nn as nn

model = resnet18(pretrained=True)


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
layers = ['layer2.0.conv1','layer2.0.conv2','layer2.1.conv1','layer2.1.conv2',
          'layer3.0.conv1','layer3.0.conv2','layer3.1.conv1','layer3.1.conv2',
          'layer4.0.conv1','layer4.0.conv2','layer4.1.conv1','layer4.1.conv2',]  

units = range(5)


#GPU
device = 'cuda:1'


#AUX 
num_workers = 4     #num workers argument in dataloader
seed = 2            #manual seed
batch_size = 1   #batch size for feeding rank image set through model (input image set is sent through all at once)

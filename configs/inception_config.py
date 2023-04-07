### Config File for Alexnet Sparse. ###


import torch
import os
import sys
from circuit_explorer import root_path

### NAME ###

name = 'inception'


###MODEL###


from lucent_video.modelzoo import inceptionv1
import torch.nn as nn

model = inceptionv1(pretrained=True)


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
layers = ['mixed4a_5x5_pre_relu_conv','mixed4a_1x1_pre_relu_conv','mixed4a_3x3_pre_relu_conv','mixed4a_pool_reduce_pre_relu_conv',
          'mixed4b_5x5_pre_relu_conv','mixed4b_1x1_pre_relu_conv','mixed4b_3x3_pre_relu_conv','mixed4b_pool_reduce_pre_relu_conv',
          'mixed4d_5x5_pre_relu_conv','mixed4d_1x1_pre_relu_conv','mixed4d_3x3_pre_relu_conv','mixed4d_pool_reduce_pre_relu_conv',
          'mixed3a_5x5_pre_relu_conv','mixed3a_1x1_pre_relu_conv','mixed3a_3x3_pre_relu_conv','mixed3a_pool_reduce_pre_relu_conv',
          'mixed5a_5x5_pre_relu_conv','mixed5a_1x1_pre_relu_conv','mixed5a_3x3_pre_relu_conv','mixed5a_pool_reduce_pre_relu_conv',
         ]

units = range(3)

#GPU
device = 'cuda:2'


#AUX 
num_workers = 4     #num workers argument in dataloader
seed = 2            #manual seed
batch_size = 1   #batch size for feeding rank image set through model (input image set is sent through all at once)

import torch
from collections import OrderedDict
from circuit_explorer.utils import load_config
from circuit_explorer.data_loading import rank_image_data
import torch.utils.data as data
import torchvision.datasets as datasets
from circuit_explorer.data_loading import rank_image_data
from circuit_explorer.target import multi_feature_target_saver
import os

config_file = '../configs/alexnet_config.py'
device = 'cuda:0'

config = load_config(config_file)
layers = config.layers
units = config.units
batch_size = 64
data_path = config.data_path
model = config.model
_ = model.eval().to(device)

out_root = './original_activations/'

imageset = data_path.split('/')[-1]
if data_path[-1] == '/':
	imageset = data_path.split('/')[-2]

#dataloader
kwargs = {'num_workers': config.num_workers, 'pin_memory': True, 'sampler':None} if 'cuda' in device else {}

label_file_path =  config.label_file_path

dataloader = data.DataLoader(rank_image_data(data_path,
												config.preprocess,
												label_file_path = label_file_path,class_folders=True),
												batch_size=batch_size,
												shuffle=False,
												**kwargs)

	
targets = OrderedDict()
original_activations = OrderedDict() 
for layer in layers:
	for unit in units:
		targets[layer+':'+str(unit)] = (layer,unit)
		original_activations[layer+':'+str(unit)] = []


#we save target activations in a context that allows us to handle the annoying problem of dangling hooks
with multi_feature_target_saver(model,targets,kill_forward=True) as target_saver:
    
    #then we just run our data through the model, the target_saver will store activations for us
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)

        target_activations = target_saver(inputs)

        #the target_saver doesnt aggregate activations, it overwrites each batch, so we need to save our data
        for target in targets:
            original_activations[target].append(target_activations[target].detach().cpu().type(torch.FloatTensor))


    #turn batch-wise list into concatenated tensor
    for target in targets:
        original_activations[target] = torch.cat(original_activations[target])

if not os.path.exists(out_root+'/'+config.name+'/'+imageset):
	os.makedirs(out_root+'/'+config.name+'/'+imageset,exist_ok=True)
torch.save(original_activations,'%s/%s/%s/original_activations.pt'%(out_root,config.name,imageset))




import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from typing import Dict, Iterable, Callable
from collections import OrderedDict
from circuit_explorer.utils import TargetReached
from circuit_explorer.data_loading import rank_image_data
from torch.utils.data import DataLoader
from circuit_explorer.utils import convert_relu_layers
import types
from copy import deepcopy


#### Target   #####

class feature_target_saver(nn.Module):
    '''
        takes a model, and adds a target feature to save the outputs of
        layer: str name of layer, use "dict([*self.model.named_modules()])"
        unit: int or list/array the length of the out features dim of the layer (specifying coefficients)

        with feature_target_saver(model, layer_name) as target_saver:
            ... run images through model
    '''
    def __init__(self, model, layer, unit, kill_forward = True):
        super().__init__()
        self.model = model
        self.layer = layer
        self.unit = unit
        self.target_activations = None
        self.layer_name = layer
        self.layer = OrderedDict([*self.model.named_modules()])[layer]
        #self.hook = self.layer.register_forward_hook(self.get_target()) #execute on forward pass
        self.hook = None
        self.kill_forward = kill_forward

    def __enter__(self, *args): 
        if self.hook is not None:
            self.hook.remove()
        self.hook = self.layer.register_forward_hook(self.get_target())       
        return self

    def __exit__(self, *args): 
        if self.hook is not None:
            self.hook.remove()
            self.hook = None

    def get_target(self) -> Callable:
        def fn(module, input, output):  #register_hook expects to recieve a function with arguments like this
            #output is what is return by the layer with dim (batch_dim x out_dim), sum across the batch dim
            if isinstance(self.unit,int):
                target_activations = output[:,self.unit]
            else:
                assert len(self.unit) == output.shape[1]
                self.unit = torch.tensor(self.unit).to(output.device).type(output.dtype)
                target_activations = torch.tensordot(output, self.unit, dims=([1],[0]))

            self.target_activations = target_activations

            #import pdb; pdb.set_trace()

            if self.kill_forward:
                #print('feature target in %s reached.'%self.layer)
                raise TargetReached
        return fn

    def forward(self, x):
        try:
            _ = self.model(x)
        except TargetReached:
            pass
        return self.target_activations


class multi_feature_target_saver(nn.Module):
    '''
        takes a model, and adds a target feature to save the outputs of
        targets: a dictionary whos values are tuples of (layer_name,unit)
            (see feature_target_saver class for conventions of layer_name/unit)

        with multi_feature_target_saver(model, layer_name) as target_saver:
            ... run images through model
    '''
    def __init__(self, model, targets, kill_forward = True):
        super().__init__()
        self.model = model
        convert_relu_layers(model)
        self.targets = targets
        self.target_activations = {}
        #self.layer = OrderedDict([*self.model.named_modules()])[layer]
        #self.hook = self.layer.register_forward_hook(self.get_target()) #execute on forward pass
        self.hooks = {}
        self.hooks_called = {}  #works in conjunction with kill_forward
        self.kill_forward = kill_forward
        self.device = next(model.parameters()).device  


    def hook_layers(self):        
        self.remove_hooks()
        for target_name, target in self.targets.items():
            layer = dict([*self.model.named_modules()])[target[0]]
            self.hooks[target_name] = layer.register_forward_hook(self.get_target(target_name))
            self.hooks_called[target_name] = False

    def remove_hooks(self):
        for target_name, target in self.targets.items():
            if target_name in self.hooks:
                self.hooks[target_name].remove()
                del self.hooks[target_name]
    
    def __enter__(self, *args): 
        self.hook_layers()
        return self

    def __exit__(self, *args): 
        self.remove_hooks()

    def get_target(self,target_name) -> Callable:
        def fn(module, input, output):  #register_hook expects to recieve a function with arguments like this
            #output is what is return by the layer with dim (batch_dim x out_dim), sum across the batch dim
            unit = self.targets[target_name][1]
            if isinstance(unit,int):
                target_activations = output[:,unit]
            else:
                assert len(unit) == output.shape[1]
                unit = torch.tensor(unit).to(output.device).type(output.dtype)
                target_activations = torch.tensordot(output, unit, dims=([1],[0]))

            self.target_activations[target_name] = target_activations

            if self.device is not None:
                self.target_activations[target_name] = self.target_activations[target_name].detach().to(self.device)

            self.hooks_called[target_name] = True

            if self.kill_forward:
                kill = True
                for t_name in self.hooks_called:
                    if not self.hooks_called[t_name]:
                        kill = False
                        break
                if kill:
                    raise TargetReached

        return fn

    def forward(self, x):
        for target_name, target in self.targets.items():
            self.hooks_called[target_name] = False
        try:
            _ = self.model(x)
        except TargetReached:
            pass
        return self.target_activations




class layer_saver(nn.Module):
    '''
        layer_saver class that allows you to retain outputs of any layer.
        This class uses PyTorch's "forward hooks", which let you insert a function
        that takes the input and output of a module as arguements.
        In this hook function you can insert tasks like storing the intermediate values,
        or as we'll do in the FeatureEditor class, actually modify the outputs.
        Adding these hooks can cause headaches if you don't "remove" them 
        after you are done with them. For this reason, the FeatureExtractor is 
        setup to be used as a context, which sets up the hooks when
        you enter the context, and removes them when you leave:
        with layer_saver(model, layer_name) as layer_saver:
            features = layer_saver(imgs)
        If there's an error in that context (or you cancel the operation),
        the __exit__ function of the feature extractor is executed,
        which we've setup to remove the hooks. This will save you 
        headaches during debugging/development.
    '''    
    def __init__(self, model, layers, retain=True, detach=True, clone=True):
        super().__init__()
        layers = [layers] if isinstance(layers, str) else layers
        self.model = model
        self.layers = layers
        self.detach = detach
        self.clone = clone
        self.device = next(model.parameters()).device 
        self.retain = retain
        self._features = {layer: torch.empty(0) for layer in layers}        
        self.hooks = {}
        
    def hook_layers(self):        
        self.remove_hooks()
        for layer_id in self.layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            self.hooks[layer_id] = layer.register_forward_hook(self.save_outputs_hook(layer_id))
    
    def remove_hooks(self):
        for layer_id in self.layers:
            if self.retain == False:
                self._features[layer_id] = torch.empty(0)
            if layer_id in self.hooks:
                self.hooks[layer_id].remove()
                del self.hooks[layer_id]
    
    def __enter__(self, *args): 
        self.hook_layers()
        return self
    
    def __exit__(self, *args): 
        self.remove_hooks()
            
    def save_outputs_hook(self, layer_id):
        def detach(output):
            if isinstance(output, tuple): return tuple([o.detach() for o in output])
            elif isinstance(output, list): return [o.detach() for o in output]
            else: return output.detach()
        def clone(output):
            if isinstance(output, tuple): return tuple([o.clone() for o in output])
            elif isinstance(output, list): return [o.clone() for o in output]
            else: return output.clone()
        def to_device(output, device):
            if isinstance(output, tuple): return tuple([o.to(device) for o in output])
            elif isinstance(output, list): return [o.to(device) for o in output]
            else: return output.to(device)
        def fn(_, __, output):
            if self.detach: output = detach(output)
            if self.clone: output = clone(output)
            if self.device: output = to_device(output, self.device)
            self._features[layer_id] = output
        return fn

    def forward(self, x):
        _ = self.model(x)
        return self._features
    


def layer_activations_from_dataloader(layers,dataloader,model,batch_size=64):
  '''
  dataloader: can be a pytorch dataloader or simply a path to an folder with images and no subfolders
  layers: should be a single layer name or list of layer names, for keys in dict "layers = OrderedDict([*model.named_modules()])"
  model: a pytorch nn model, set the device of model to determine devices for all variables in this function
  '''

  device = next(model.parameters()).device

  #generate dataloader if image_path passed
  if isinstance(dataloader,str):
    kwargs = {'num_workers': 4, 'pin_memory': True, 'sampler':None} if 'cuda' in device.type else {}
    dataloader = DataLoader(rank_image_data(dataloader,
                                            class_folders=False,
                                            rgb=True),
                                            batch_size=batch_size,
                                            shuffle=False,
                                            **kwargs)
  

  layer_activations = {}
  if isinstance(layers,str):
    layers = [layers]
  for i in layers:
    layer_activations[i] = []
  

  for i, data in enumerate(dataloader):
    #if i%int(len(dataloader)/4) == 0:
    #  print(str(i)+'/'+str(len(dataloader)))
    images = data[0].to(device)
    with layer_saver(model, layers) as extractor:
      batch_layer_activations = extractor(images) #all features for layer and all images in batch
      for i in layers:
        layer_activations[i].append(batch_layer_activations[i].detach().to('cpu'))

  for i in layers:     
    layer_activations[i] = torch.cat(layer_activations[i])

  return layer_activations
    




### Losses ### 
'''
The feature target saver saves something multidimensional, but we have to 
backprop 'losses', basically scalars. Here are different reasonable functions
for collapsing a target feature into a scalar, that can be back propogated.
'''

def sum_loss(target):
    return torch.sum(target)

def sum_abs_loss(target):
    return torch.sum(torch.abs(target))

# def positional_loss(target,position):
#     #position should be (batch_i,H,W)
#     return target[position[0],position[1],position[2]]

def pearson_loss(x,y):
    return x * y * torch.rsqrt(torch.sum(x ** 2)) * torch.rsqrt(torch.sum(y ** 2))

class positional_loss(nn.Module):
    '''
    position should be (H,W)
    target should be (batch,H,W) (channel already selected)
    '''
    def __init__(self, position, loss_func = sum_abs_loss):
        super().__init__()
        self.position = position
        self.loss_func = loss_func

    def forward(self,target):
        #return target[:,self.position[0],self.position[1]].mean(dim=0)
        return self.loss_func(target[:,self.position[0],self.position[1]])
    
class distance_2_target_loss(nn.Module):
    '''
    position should be (H,W)
    target should be (batch,H,W) (channel already selected)
    '''
    def __init__(self, target, loss_func = nn.L1Loss()):
        super().__init__()
        self.target = target
        self.loss_func = loss_func

    def forward(self,x):
        #return target[:,self.position[0],self.position[1]].mean(dim=0)
        return self.loss_func(x,self.target)
    

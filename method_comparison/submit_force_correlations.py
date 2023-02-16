from subprocess import call
from circuit_pruner.utils import load_config

config_file = '../configs/vgg11_config.py'

config = load_config(config_file)
layers = config.layers
units = config.units
batch_size = 64


print(batch_size)
print(layers)
print(units)

data_path = '../image_data/imagenet_2/'
device = 'cuda:0'

#sparsities = [.9,.8,.7,.6,.5,.4,.3,.2,.1,.05,.01,.005,.001]
sparsities = [.2]
layers = [layers[0]]
units = [units[0]]

out_root = '/mnt/data/chris/nodropbox/Projects/circuit_pruner/correlations/'

original_act_file = '/mnt/data/chris/nodropbox/Projects/circuit_pruner/original_activations/%s/imagenet_2/original_activations.pt'%config.name

del config

for unit in units:
    print('PROCESSING UNIT: %s'%str(unit))
    for layer in layers:
        print('PROCESSING LAYER: %s'%layer)
        for sparsity in sparsities:
            call_str = 'python force_correlation.py --unit %s --layer %s --sparsity %s --config %s --data-path %s --device %s --out-root %s --batch-size %s --original_act_file %s'%(str(unit),layer,str(sparsity),config_file,data_path,device,out_root,str(batch_size),original_act_file)
            call(call_str,shell=True)
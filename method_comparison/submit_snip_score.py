from subprocess import call
from circuit_explorer.utils import load_config

config_file = '../configs/alexnet_sparse_config.py'

config = load_config(config_file)
layers = config.layers
units = config.units
batch_size = config.batch_size

del config

print(batch_size)
print(layers)
print(units)

data_path = '../image_data/imagenet_2/'
device = 'cuda:1'

out_root = './circuit_scores/'



for unit in units:
    print('PROCESSING UNIT: %s'%str(unit))
    for layer in layers:
        print('PROCESSING LAYER: %s'%layer)
        call_str = 'python snip_score.py --unit %s --layer %s --config %s --data-path %s --device %s --out-root %s --batch-size %s'%(str(unit),layer,config_file,data_path,device,out_root,str(batch_size))
        call(call_str,shell=True)

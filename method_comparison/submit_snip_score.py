from subprocess import call
from circuit_pruner.utils import load_config

config_file = '../configs/vgg11_config.py'

config = load_config(config_file)
layers = config.layers
units = config.units
batch_size = config.batch_size

del config

print(batch_size)
print(layers)
print(units)

data_path = '../image_data/imagenet_2/'
device = 'cuda:0'

out_root = '/mnt/data/chris/nodropbox/Projects/circuit_pruner/circuit_ranks/'



for unit in units:
    print('PROCESSING UNIT: %s'%str(unit))
    for layer in layers:
        print('PROCESSING LAYER: %s'%layer)
        call_str = 'python snip_score.py --unit %s --layer %s --config %s --data-path %s --device %s --out-root %s --batch-size %s'%(str(unit),layer,config_file,data_path,device,out_root,str(batch_size))
        call(call_str,shell=True)
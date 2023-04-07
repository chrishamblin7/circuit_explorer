#quick script for downloading large files stored on google drive

import requests
from subprocess import call
import os
from circuit_explorer import root_path

#Dictionary with the online gdrive keys for models image folder etc.
online_models = {'mnist':{'model':['1X6wR6nJ_SguVzd6MVFelvXsH9G2uR4WZ','mnist_statedict.pt'],'images':['1rGXi_pWGvz3UsdO1FpkWc2WsU-M42Z3v','mnist']},
				 'cifar10':{'model':None,'images':['17pjtPG-MJK7mhTh_KHvHLHUwSButkwLA','cifar10']},
				 'alexnet':{'model':None,'images':['1Onu6pUVH4m0GNHFgc7rTmOfLHQzCUdJY','imagenet_2']},
				 'alexnet_sparse':{'model':['1MMr2LgwQkQIDb8SNwqLaqnezXJOVUHps','alexnet_sparse_statedict.pt'],'images':['1Onu6pUVH4m0GNHFgc7rTmOfLHQzCUdJY','imagenet_2']},
				}

online_model_names = list(online_models.keys())

def file_download_old(id, destination):
	URL = "https://docs.google.com/uc?export=download"

	session = requests.Session()

	response = session.get(URL, params = { 'id' : id }, stream = True)
	token = get_confirm_token(response)

	if token:
		params = { 'id' : id, 'confirm' : token }
		response = session.get(URL, params = params, stream = True)

	save_response_content(response, destination)    

def get_confirm_token(response):
	for key, value in response.cookies.items():
		if key.startswith('download_warning'):
			return value

	return None

def save_response_content(response, destination):
	CHUNK_SIZE = 32768

	with open(destination, "wb") as f:
		for chunk in response.iter_content(CHUNK_SIZE):
			if chunk: # filter out keep-alive new chunks
				f.write(chunk)

def file_download(id,destination):
	call('''wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=%s' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=%s" -O %s && rm -rf /tmp/cookies.txt '''%(id,id,destination),shell=True)


def tar_download(id,dest_path):
	print('downloading')
	file_download(id,dest_path)
	print('untaring')
	out_dir = '/'.join(dest_path.split('/')[:-1])
	call('tar -xzvf %s -C %s'%(dest_path,out_dir),shell=True)
	call('rm %s'%dest_path,shell=True)	


def download_from_gdrive(model,target = 'model'):
	#model key from online_models duct at the top of this script
	#target is from ['model','images']

	#make directories if dont exist
	if not os.path.exists(root_path+'/image_data'):
		os.mkdir(root_path+'/image_data')
	if not os.path.exists(root_path+'/models'):
		os.mkdir(root_path+'/models')


	if target == 'model':
		if not os.path.exists(root_path+'/models/%s'%online_models[model]['model'][1]):
			print('Downloading model')
			file_download(online_models[model]['model'][0],root_path+'/models/%s'%online_models[model]['model'][1])
		else:
			print('not downloading model %s, as it already exists in "models" folder.'%online_models[model]['model'][1])

	elif target == 'images':
		if not os.path.exists(root_path+'/image_data/%s'%online_models[model]['images'][1]):
			print('Downloading input image data associated with %s: %s\n\n'%(model,online_models[model]['images'][1]))
			tar_download(online_models[model]['images'][0],root_path+'/image_data/%s.tgz'%model)
		else:
			print('not downloading image dataset %s, as it already exists in "image_data"'%online_models[model]['images'][1])

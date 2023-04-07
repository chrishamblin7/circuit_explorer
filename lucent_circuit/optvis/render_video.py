import torchvision
from lucent_circuit.optvis import render, param, transform, objectives
from PIL import Image
import torch
from circuit_explorer.mask import setup_net_for_mask
from circuit_explorer.utils import convert_relu_layers
from collections import OrderedDict
import numpy as np

def render_accentuation(img_path,
                        layer,
                        unit,
                        model,
                        size=224,
                        saturation=8.,
						magic=30.,
						device = None,
                        thresholds=range(41),
						obj_f=None,
                        optimizer=None,  #lucent optimizer
                        transforms=None, #lucent transforms
                        show_image=True,
                        include_noise_init=True,
                        noise_std = .01,
						l1_reg=0
                        ):
  if device is None: device = next(model.parameters()).device
  layer = layer.replace('.','_')
  convert_relu_layers(model)
  mean = .5  #mean and standard deviation derived from synthetic images
  std = .08
  transformation = torchvision.transforms.Compose([torchvision.transforms.Resize((size,size)),torchvision.transforms.ToTensor()])
  img = Image.open(img_path)
  img = img.convert("RGB") 
  im_tensor = transformation(img).float().unsqueeze(0)
  im_tensor = im_tensor.to(device)
  im_tensor = ((im_tensor-im_tensor.mean())/im_tensor.std())*std+mean
  fourier_image = param.spatial.image_2_fourier(im_tensor, saturation=saturation)

  fourier_image.requires_grad_(False)
  #double it
  fourier_image = torch.cat((fourier_image,fourier_image),dim=0)
  fourier_image.requires_grad_(True)
  param_f = lambda: param.image(size,start_params=fourier_image,fft=True,device=device,batch=2)
  if obj_f is None:
    obj_f = objectives.neuron
    if hasattr(unit,'__len__'):
      obj_f = objectives.dotdirection_neuron
  obj = obj_f(layer,unit,batch=0) - obj_f(layer,unit,batch=1) - l1_reg*objectives.l1_distance("input",.5,batch=0) - l1_reg*objectives.l1_distance("input",.5,batch=1)
  if include_noise_init:
    noise_param_f = lambda: param.image(size,device=device,sd=noise_std,magic=magic)
    noise_params, noise_image_f = noise_param_f()
    noise_params = noise_params[0].requires_grad_(False)
    noise_params = torch.cat((noise_params,noise_params),dim=0)
    fourier_image.requires_grad_(False)
    #fourier_image = torch.cat((fourier_image,(fourier_image-torch.mean(fourier_image))/torch.std(fourier_image)*noise_std),dim=0)
    fourier_image = torch.cat((fourier_image,noise_params),dim=0)
    fourier_image.requires_grad_(True)
    param_f = lambda: param.image(size,start_params=fourier_image,fft=True,device=device,batch=4)
    obj = obj_f(layer,unit,batch=0) - obj_f(layer,unit,batch=1) + obj_f(layer,unit,batch=2) - obj_f(layer,unit,batch=3) - l1_reg*objectives.l1_distance("input",.5,batch=0) - l1_reg*objectives.l1_distance("input",.5,batch=1)
  show_inline = False
  if show_image: show_inline =True
  output = render.render_vis(model, obj, param_f,optimizer=optimizer,transforms=transforms, thresholds=thresholds,progress=True,show_image=show_image,show_inline=show_inline,device=device)
  return output

def render_video(model,scores,layer,unit,
						  sparsities = None,sparsity_range = (.001,.6), frames=200,
						  scheduler='linear',connected=True,opt_steps=50,lr = 5e-2,size=224,
						  desaturation=10,file_name=None,save=True, negatives=True, include_full= True, reverse=False):
	
	setup_net_for_mask(model)
	
	if negatives:
		batch_size=4
		obj = objectives.channel(layer, unit, batch=0) + objectives.neuron(layer, unit, batch=1) - objectives.channel(layer, unit, batch=2) - objectives.neuron(layer, unit, batch=3)
	else:
		batch_size=2
		obj = objectives.channel(layer, unit, batch=0) + objectives.neuron(layer, unit, batch=1)

	image_list = []
	params = None
	opt = lambda params: torch.optim.Adam(params, lr)
	
	
	
	nonzero_scores = 0 
	for layer in scores:
		
		nonzero_scores += torch.sum((scores[layer] != 0).int())
	ks = []
	
	
	if sparsities is None:
		if scheduler == 'exp':

			max_k = int(nonzero_scores*sparsity_range[1])
			min_k = int(nonzero_scores*sparsity_range[0])
			for t in range(0,frames+1):
				k = ceil(exp(t/frames*log(min_k)+(1-t/frames)*log(max_k))) #exponential schedulr
				ks.insert(0, k) 
		else:
			sparsities = np.linspace(sparsity_range[0],sparsity_range[1],frames)
			for sparsity in sparsities:
				ks.append(int(nonzero_scores*sparsity))
	else:
		for sparsity in sparsities:
			ks.append(int(nonzero_scores*sparsity))
			
	#include_full
	if include_full and ks[-1] < nonzero_scores:
		ks.append(int(nonzero_scores))

	ks = list(OrderedDict.fromkeys(ks)) #remove duplicates
	for j in range(5):
		if j in ks:
			ks.remove(j)

	if reverse:
		ks.reverse()
	
	
	for k in ks:
		start = time.time()
		print(k)
		mask = mask_from_scores(scores,num_params_to_keep=k)
		#mask,cum_sal = mask_from_sparsity(ranks,k)
		#expanded_mask = expand_structured_mask(mask,model)
		apply_mask(model,mask)
		if connected:
			if params is not None:
				params = params.requires_grad_(False)
				##add noise
				#param_noise, _ = param.fft_image((1, 3, size, size),sd=.01)
				#param_noise = param_noise[0].requires_grad_(False)
				#params = param_noise+params
				
				#desaturate
				params = (params-torch.mean(params))/torch.std(params)*.05
				
				
				##remove high frequency 
				#params[:,:,:,10:,:] = 0
				
				params = params.requires_grad_(True)
				

			param_f = lambda: param.image(size,start_params=params,magic=desaturation,batch=batch_size)
		else:
			param_f = lambda: param.image(size,batch=batch_size)
		#import pdb; pdb.set_trace()
		output = render.render_vis(model, obj, param_f, opt, thresholds=(opt_steps,),progress=True,show_inline=True)
		image = output['images']
		params= output['params']
		
		image_list.append(image[0])    #####[0] MIGHT BE UNNECSSARY
		
		#include full
		if include_full and (k ==ks[-1]) and not reverse:
			for _ in range(5):
				image_list.append(image[0])
				
		print(time.time()-start)

	#reshape
	for i in range(len(image_list)):
		im = image_list[i]
		if im.shape[0] > 1:
			image_list[i] = np.concatenate(tuple(im),axis=1)

	gif = []
	for image in image_list:
		im = Image.fromarray(np.uint8(image*255))
		gif.append(im)
	   
	if file_name is None:
		connected_name = ''
		if connected:
			connected_name = 'connected'
			file_name = '%s_%s_%s_paint.gif'%(layer,unit,connected_name)
		
	if save:
		print('saving gif to %s'%file_name)
		gif[0].save(file_name, save_all=True,optimize=False, append_images=gif[1:], loop=0)
	
	return image_list

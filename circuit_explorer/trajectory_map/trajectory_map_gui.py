#loading base packages
import os
import torch
from PIL import Image
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torchvision
import pandas as pd
import pickle
from copy import deepcopy
from math import ceil,floor

#library specific packages
from circuit_explorer.target import sum_abs_loss, positional_loss, feature_target_saver
from circuit_explorer.mask import setup_net_for_mask, mask_from_scores, apply_mask
from circuit_explorer.utils import params_2_target_from_scores, convert_relu_layers
from circuit_explorer.score import actgrad_filter_score, actgrad_kernel_score, get_num_params_from_cum_score, snip_score
from circuit_explorer.data_loading import single_image_data, default_preprocess
from lucent_circuit.optvis import render, param, transform, objectives
from lucent_circuit.optvis.render_video import render_accentuation
from circuit_explorer.dissected_Conv2d import dissect_model
#make relus not inplace, important visualizing negative activations for example
from circuit_explorer.receptive_fields import position_crop_image,receptive_field

#plotting
import plotly.express as px
from jupyter_dash import JupyterDash
from dash import dcc, html, Input, Output, no_update, State, ctx
import plotly.graph_objects as go
import io
import base64
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix

from circuit_explorer.utils import color_string_to_list, value_to_color_from_cscale

default_input_size = (3,224,224)

def add_border_to_PIL_image(img,c,cscale,cmin,cmax,ratio=10):
    size = img.size
    color = value_to_color_from_cscale(c,cscale,cmin,cmax)
    new_size = (size[0]+int(size[0]/ratio), size[1]+int(size[1]/ratio))
    new_img = Image.new("RGB", new_size, color = tuple(color))   ## luckily, this is already black!
    box = tuple((n - o) // 2 for n, o in zip(new_size, size))
    new_img.paste(img, box)
    return new_img


def image_path_to_base64(im_path,layer,pos=None,size = (default_input_size[1],default_input_size[2]),rf_dict=None,boundary_data=None):
    img = Image.open(im_path)
    if pos is not None:
        img = position_crop_image(img,pos,layer,rf_dict=rf_dict)
    else:
        img = img.resize(size)
    if boundary_data is not None:
        img = add_border_to_PIL_image(img,
                                    boundary_data['c'],
                                    boundary_data['cscale'],
                                    boundary_data['cmin'],
                                    boundary_data['cmax'])
        
    buffer = io.BytesIO()
    img.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url


def pil_image_to_base64(img,layer,pos=None,size = (default_input_size[1],default_input_size[2]),rf_dict=None,boundary_data=None):
    if pos is not None:
        img = position_crop_image(img,pos,layer,rf_dict=rf_dict)
    else:
        img = img.resize(size)
    if boundary_data is not None:
        img = add_border_to_PIL_image(img,
                                    boundary_data['c'],
                                    boundary_data['cscale'],
                                    boundary_data['cmin'],
                                    boundary_data['cmax'])
    buffer = io.BytesIO()
    img.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url


def umap_fig_from_df(df,data_folder=None,layer=None,normed=False,norm_column = 'l1_norm',align_df=None,num_display_images=50,act_column = None,color_std=None,show_colorscale=True,rf_dict=None,image_boundary=True):
    '''
    df: a umap df
    data_folder: path to images
    align_df: df to rotationally align to (usually done before hand)
    '''
    fig = go.Figure()
    #positions
    xy_addition = ''
    if normed:
        xy_addition = '_normed'
    x = list(df['x'+xy_addition])
    y = list(df['y'+xy_addition])
    #norms
    norms = list(df[norm_column])
    #activations
    act_column = act_column or 'activation'
    acts = list(df[act_column]) 
    
    #color
    color_std = color_std or torch.std(torch.tensor(acts)) 
    color_limit = float(color_std*3)

    fig.add_trace(go.Scatter(
      x=x,
      y=y,
      marker=dict(
                  line=dict(width=.5,
                            color='Grey'),
                  cmid=0,
                  cmin=-color_limit,
                  cmax=color_limit,
                  size=torch.tensor(norms)/torch.mean(torch.tensor(norms))*3.5,
                  color=acts,
                  colorbar=dict(
                                  title="Activation"
                              ),
                  colorscale="RdBu_r"
                  ),
      mode="markers",
      name='points'
      ))
    
    
    #alignment trace
    if align_df is not None:
        #add line figs
        x_align = list(align_df['x'+xy_addition])
        y_align = list(align_df['y'+xy_addition])
        x_joint = []
        y_joint = []
        for i in range(len(x)):
            x_joint.append(x_align[i])
            x_joint.append(x[i])
            x_joint.append(None)
            y_joint.append(y_align[i])
            y_joint.append(y[i])
            y_joint.append(None)

        fig.add_trace(go.Scatter(
                                x=x_joint, 
                                y=y_joint,
                                line=dict(color='grey', width=.5),
                                mode='lines',
                                name='alignment',
                                visible='legendonly'))

        
    if not show_colorscale:
        fig.update_traces(marker_showscale=False)
    
    layout = go.Layout(   margin = dict(l=10,r=10,b=10,t=10),
                          legend=dict(yanchor="top", xanchor="left", x=0.0),
                          paper_bgcolor='rgba(255,255,255,1)',
                          plot_bgcolor='rgba(255,255,255,1)',
                          xaxis=dict(showline=False,showgrid=False,showticklabels=False,range=[torch.min(torch.tensor(x))-1, torch.max(torch.tensor(x))+1]),
                          yaxis=dict(showline=False,showgrid=False,showticklabels=False,scaleanchor="x", scaleratio=1,range=[torch.min(torch.tensor(y))-1, torch.max(torch.tensor(y))+1]))

    fig.layout = layout
    

    #images
    if (data_folder is not None) and num_display_images>0:
        
        #select images far apart
        pts2D = np.swapaxes(np.array([list(df['x']),list(df['y'])]),0,1)
        kmeans = KMeans(n_clusters=num_display_images, random_state=0).fit(pts2D)
        labels = kmeans.predict(pts2D)
        cntr = kmeans.cluster_centers_
        approx = []
        for i, c in enumerate(cntr):
            lab = np.where(labels == i)[0]
            pts = pts2D[lab]
            d = distance_matrix(c[None, ...], pts)
            idx1 = np.argmin(d, axis=1) + 1
            idx2 = np.searchsorted(np.cumsum(labels == i), idx1)[0]
            approx.append(idx2)
        #add layout images
        use_position=None
        if 'position' in df.columns:
            use_position = True
        for i in approx:
            boundary_data=None
            if image_boundary:
                boundary_data = {'c':acts[i],
                                 'cscale':fig.data[0]['marker']['colorscale'],
                                 'cmin':-color_limit,
                                 'cmax':color_limit}
            position=None
            if use_position:
                position = df.iloc[i]['position']  
            img = image_path_to_base64(data_folder+'/'+df.iloc[i]['image'],
                                       layer,pos=position,
                                       size = (default_input_size[1],default_input_size[2]),
                                       rf_dict=rf_dict,boundary_data=boundary_data)
            #img = Image.open(data_folder+'/'+df.iloc[i]['image']) 
            fig.add_layout_image(
                                dict(
                                    source=img,
                                    #source="http://chrishamblin.xyz/images/viscnn_images/%s.jpg"%nodeid,
                                    xref="x",
                                    yref="y",
                                    x=df.iloc[i]['x'+xy_addition],
                                    y=df.iloc[i]['y'+xy_addition],
                                    sizex=.5,
                                    sizey=.5,
                                    xanchor="center",
                                    yanchor="middle",
                                    layer='above',
                                    opacity=.5
                                ))
    

    # turn off native plotly.js hover effects - make sure to use
    # hoverinfo="none" rather than "skip" which also halts events.
    fig.update_traces(hoverinfo="none", hovertemplate=None)

    return fig



def full_app_from_df(df,data_folder,model,layer,unit,normed=False,norm_column='l1_norm', align_df = None,max_images=200,image_order=None,use_kernels=True,preprocess=default_preprocess,input_size=default_input_size,image_boundary=True):
    device = next(model.parameters()).device 
    convert_relu_layers(model)

    rf_dict = receptive_field(model, input_size, print_output=False)

    use_position = False
    if 'position' in df.columns:
        use_position = True

    if use_kernels:
        dis_model = dissect_model(deepcopy(model))  
        _ = dis_model.eval().to(device)
        pruning_type_selector = dcc.RadioItems(['filters', 'kernels', 'weights'],'filters',id='pruning_type')
    else:
        pruning_type_selector = dcc.RadioItems(['filters', 'weights'],'filters',id='pruning_type')
    
    top_row = df[df['activation'] == df['activation'].max()].iloc[0]
    top_row_pos = None
    if use_position:
        top_row_pos = top_row['position']
    start_image = image_path_to_base64(data_folder+top_row['image'],layer,pos=top_row_pos,rf_dict=rf_dict)

    umap_fig = umap_fig_from_df(df,layer=layer,normed=normed, align_df=align_df,rf_dict=rf_dict,norm_column=norm_column)
    
    xy_addition = ''
    if normed:
        xy_addition = '_normed'
    
    if image_order is None:
        #image order
        #select images far apart
        pts2D = np.swapaxes(np.array([list(df['x']),list(df['y'])]),0,1)
        kmeans = KMeans(n_clusters=max_images, random_state=0).fit(pts2D)
        labels = kmeans.predict(pts2D)
        cntr = kmeans.cluster_centers_
        image_order = []
        for i, c in enumerate(cntr):
            lab = np.where(labels == i)[0]
            pts = pts2D[lab]
            d = distance_matrix(c[None, ...], pts)
            idx1 = np.argmin(d, axis=1) + 1
            idx2 = np.searchsorted(np.cumsum(labels == i), idx1)[0]
            image_order.append(idx2)
    #all layout images
    all_layout_images = []        
    for i in image_order:
        position=None
        boundary_data=None
        if image_boundary:
            boundary_data = {'c':float(df.iloc[i]['activation']),
                             'cscale':umap_fig.data[0]['marker']['colorscale'],
                             'cmin':umap_fig.data[0]['marker']['cmin'],
                             'cmax':umap_fig.data[0]['marker']['cmax']}
        if use_position:
            position = df.iloc[i]['position']  
        img = image_path_to_base64(data_folder+'/'+df.iloc[i]['image'],
                                    layer,pos=position,
                                    size = (default_input_size[1],default_input_size[2]),
                                    rf_dict=rf_dict,boundary_data=boundary_data)
        #img = Image.open(data_folder+'/'+df.iloc[i]['image']) 
        all_layout_images.append(
                                    dict(
                                        source=img,
                                        #source="http://chrishamblin.xyz/images/viscnn_images/%s.jpg"%nodeid,
                                        xref="x",
                                        yref="y",
                                        x=df.iloc[i]['x'+xy_addition],
                                        y=df.iloc[i]['y'+xy_addition],
                                        sizex=.5,
                                        sizey=.5,
                                        xanchor="center",
                                        yanchor="middle",
                                        layer='above',
                                        opacity=.8
                                    ))


    #external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    #app = JupyterDash(__name__,external_stylesheets = external_stylesheets)
    app = JupyterDash(__name__)

    colors = {
            'background': 'rgba(255,255,255,1)',
            'text': '#111111'
            }

    app.layout = html.Div([

      html.Div([
        #dcc.Graph(id="umap", figure=umap_fig, clear_on_unhover=True,style={'width': '100vh', 'height': '100vh'}),
        dcc.Graph(id="umap", figure=umap_fig, clear_on_unhover=True),
        html.Label('# plot images'),
        dcc.Slider(0, len(image_order), 1,
                  marks={i: str(i) for i in range(0,len(image_order),int(len(image_order)/10))},
                  value=0,
                  id='num_images_slider'
                 ),
        html.Label('size plot images'),
        dcc.Slider(.001, 1, .01,
                  marks={i/10: str(round(i/10,1)) for i in range(0,10)},
                  value=.5,
                  id='size_images_slider'
                 ),
        dcc.Tooltip(id="graph-tooltip")
               ],style={'width': '49%','display': 'inline-block'}),
      html.Br(),
        
      html.Div([
        html.Img(src=start_image, id='click-image'),
        html.Img(src=start_image, id='accent-image-max'),
        html.Img(src=start_image, id='noise-image-max'),
        html.Img(src=start_image, id='accent-image-min'),
        html.Img(src=start_image, id='noise-image-min'),
        html.Br()
                ],style={'width': '90%','display': 'inline-block'}),
        
      html.Div([
        html.Label('accentuation steps'),
        dcc.Slider(0, 39, 1,
                  value=20,
                  marks={i: str(i) for i in range(0,40,5)},
                  id='accent_threshold_slider'
                 ),
        html.Label('saturation'),
        dcc.Slider(0, 1, .01,
                  value=.5,
                  marks={i*.1: '{}'.format(round(i*.1,1)) for i in range(10)},
                  id='saturation_slider'
                 ),
        html.Label('cumulative weight in model'),
        dcc.Slider(.5, 1, .005,
                  value=.98,
                  marks={.5+ i*.05: '{}'.format(round(.5+ i*.05,2)) for i in range(20)},
                  updatemode='drag',
                  id='accent_sparsity_slider',
                 ),
        html.Br(),
        html.Label('pruning type'),
        pruning_type_selector,
        html.Div(id='sparsity'),
      ],style={'width': '49%','display': 'inline-block'}),
        
      #background
      dcc.Store(id='memory')
      ],style={'backgroundColor':colors['background'],'color':colors['text']})


    @app.callback(
      Output("graph-tooltip", "show"),
      Output("graph-tooltip", "bbox"),
      Output("graph-tooltip", "children"),
      Input("umap", "hoverData"),
    )
    def display_hover(hoverData):
        if hoverData is None:
            return False, no_update, no_update

        # demo only shows the first point, but other points may also be available
        pt = hoverData["points"][0]
        bbox = pt["bbox"]
        num = pt["pointNumber"]
        df_row = df.iloc[num]
        
        position=None
        if use_position:
            position = df_row['position']
        
        img_path = data_folder+'/'+df_row['image']
        img_src = image_path_to_base64(img_path,layer,pos=position,rf_dict=rf_dict)
        act = round(df_row['activation'],3)
        norm = round(df_row[norm_column],3)

        children = [
          html.Div(children=[
                              html.Img(src=img_src, style={"width": "100%"}),
                              html.P(f"activation: {act}"),
                              html.P(f"norm: {norm}"),
                              html.P(f"Image: {df_row['image']}"),
                              html.P(f"Position: {position}")
                              ],style={'width': '200px', 'white-space': 'normal'})
                    ]

        return True, bbox, children


    @app.callback(
    Output("memory", "data"),
    Input("umap", "clickData"),
    Input("accent_sparsity_slider",'value'),
    Input("saturation_slider",'value'),
    Input("pruning_type","value")
    )
    def store_images(clickData,cum_score,sat,pruning_type):

        # demo only shows the first point, but other points may also be available
        pt = clickData["points"][0]
        num = pt["pointNumber"]

        df_row = df.iloc[num]
        img_path = data_folder+'/'+df_row['image']
        
        position=None
        loss = sum_abs_loss
        if use_position:
            position = df_row['position']
            loss = positional_loss(position)


        #get scores
        setup_net_for_mask(model)
        
        sparsity = 1.
        if cum_score < 1:
            dataloader = DataLoader(single_image_data(img_path,
                                                      preprocess),
                                    batch_size=1,
                                    shuffle=False
                                    )
            if pruning_type == 'weights':
                scores = snip_score(model,dataloader,layer.replace('_','.'),unit,loss_f=loss)
            elif pruning_type == 'kernels':
                scores = actgrad_kernel_score(dis_model,dataloader,layer.replace('_','.'),unit,loss_f=loss,dissect_model=False)
            else:
                scores = actgrad_filter_score(model,dataloader,layer.replace('_','.'),unit,loss_f=loss)  
            keep_params = get_num_params_from_cum_score(scores,cum_score)
            total_params = params_2_target_from_scores(scores,unit,layer,model)
            sparsity = keep_params/total_params
            mask = mask_from_scores(scores, num_params_to_keep = keep_params)
            apply_mask(model,mask,zero_absent=False) #dont zero absent as scores dont have target layer



        orig_pil_img = Image.open(img_path)
        boundary_data=None
        if image_boundary:
            boundary_data = {'c':float(df_row['activation']),
                             'cscale':umap_fig.data[0]['marker']['colorscale'],
                             'cmin':umap_fig.data[0]['marker']['cmin'],
                             'cmax':umap_fig.data[0]['marker']['cmax']}
        orig_img_src = pil_image_to_base64(orig_pil_img,layer,pos=position,rf_dict=rf_dict,boundary_data=boundary_data)
        accent_output = render_accentuation(img_path,layer.replace('.','_'),unit,model,saturation=sat*16.,device=device,size=input_size[1],show_image=False)

        data = {'images':{'orig':orig_img_src,
                        'max':{},
                        'min':{},
                        'min_noise':{},
                        'max_noise':{}},
                 'sparsity':round(sparsity,3)}

        for frame, frame_image in enumerate(accent_output['images']):
            all_accent_tensor_img = frame_image
            accent_tensor_img_max = all_accent_tensor_img[0]
            accent_tensor_img_min = all_accent_tensor_img[1]
            noise_tensor_img_max = all_accent_tensor_img[2]
            noise_tensor_img_min = all_accent_tensor_img[3]
            accent_img_max = Image.fromarray(np.uint8(accent_tensor_img_max*255))
            accent_img_min = Image.fromarray(np.uint8(accent_tensor_img_min*255))
            noise_img_max = Image.fromarray(np.uint8(noise_tensor_img_max*255))
            noise_img_min = Image.fromarray(np.uint8(noise_tensor_img_min*255))
            data['images']['max']['frame %s'%(str(frame))] = pil_image_to_base64(accent_img_max,layer,pos=position,rf_dict=rf_dict)
            data['images']['min']['frame %s'%(str(frame))] = pil_image_to_base64(accent_img_min,layer,pos=position,rf_dict=rf_dict)
            data['images']['max_noise']['frame %s'%(str(frame))] = pil_image_to_base64(noise_img_max,layer,pos=position,rf_dict=rf_dict)
            data['images']['min_noise']['frame %s'%(str(frame))] = pil_image_to_base64(noise_img_min,layer,pos=position,rf_dict=rf_dict)

        return data



    @app.callback(
      Output("click-image", "src"),
      Output('accent-image-max','src'),
      Output('accent-image-min','src'),
      Output('noise-image-max','src'),
      Output('noise-image-min','src'),
      Input("memory", "data"),
      Input("accent_threshold_slider", "value"),
    )
    def display_click_images(memory,frame):

        frame = int(frame)
        return memory['images']['orig'], memory['images']['max']['frame '+str(frame)], memory['images']['min']['frame '+str(frame)], memory['images']['max_noise']['frame '+str(frame)],memory['images']['min_noise']['frame '+str(frame)]

    
    @app.callback(
        Output('sparsity', 'children'),
        Input('memory', 'data')
    )
    def update_sparsity(memory):
        return 'Sparsity: %s'%str(memory["sparsity"])
     
        
        
    @app.callback(
                  Output("umap", "figure"),
                  Input("num_images_slider", "value"),
                  Input("size_images_slider", "value"),
                  State("umap", "figure"),
                )
    def display_fig_images(num_images,size_images,fig):
        trigger = ctx.triggered_id.split('.')[0]
        if trigger == 'size_images_slider':
            fig['layout']['images'] = list(fig['layout']['images'])
            for i,img in enumerate(fig['layout']['images']):
                fig['layout']['images'][i]['sizex'] = float(size_images)
                fig['layout']['images'][i]['sizey'] = float(size_images)
            fig['layout']['images'] = tuple(fig['layout']['images'])
            return fig
        try:
            current_images = len(fig['layout']['images'])
        except:
            current_images = 0
            fig['layout']['images'] = tuple()
        if num_images <=current_images:
            fig['layout']['images'] = fig['layout']['images'][:num_images]
            return fig
        new_images = list(fig['layout']['images'])
        for k in range(current_images,num_images):
            new_images.append(all_layout_images[k])
        fig['layout']['images'] = tuple(new_images)
        return fig
    
    return app


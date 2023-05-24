# CIRCUIT EXPLORER

## Colab Notebooks
[image-feature trajectory map](https://colab.research.google.com/drive/1WBgEmgMeh4CIjNVFQaY5cukWYogRjmRO?usp=share_link#scrollTo=IJ22BN9Cx_XM)

## Setup

Create a [virtual environment](https://realpython.com/python-virtual-environments-a-primer/) for the software.
* `python3 -m venv env`
* `source env/bin/activate`

Install dependencies: `pip install -r requirements.txt`

Add the circuit_explorer package itself: `pip install -e .`

Make the virtual environment available to jupyter notebooks:  `python -m ipykernel install --user --name=circuit_explorer`

## Tutorials with Jupyter Notebooks
 
Demo Jupyter Notebooks of experiments can be found in the folder 'demo_notebooks'. 
* `api.ipynb` : A notebook with general examples of how to use the circuit pruner API.
* `trajectory_map.ipynb` a notebook for generating and launching the UI for 'image-feature' trajectory maps
* `circuit_diagram.ipynb` : A notebook for generating 'circuit diagrams'
* `polysemantic_subcircuits.ipynb` : A notebook for the polysemantic subcircuit experiment
* `circle_subcircuits.ipynb` : A notebook for the circle subcircuit experiment

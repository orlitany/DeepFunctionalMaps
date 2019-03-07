# Deep Functional Maps
This page contains a TensorFlow implementation (version 1.3.0) of the method described in https://arxiv.org/pdf/1704.08686

![Alt text](/fmnet.png?raw=true "Teaser")


## Instructions
* Download Data and Results folders from: https://drive.google.com/drive/folders/1nHUGaKcn2INwXln6Ig354y6o9WI-_Kzj?usp=sharing
* To run on a pair of shapes, update their filenames in test_pairs.txt
* run test_FMnet.py

## Data description
The network recieves as input a pair of shapes, in the format of a mat struct with precomputed shot descriptors and Laplacian eigenfunctions. See the example shapes provided in './Data/'.

### Data pre-processing
Faust models are scaled by a factor of 100. To compute SHOT descriptors, the calc_shot function was used (see Utils folder) with the following parameters: num_bins = 10, radius = 9, min_neighs = 3:
calc_shot([model.X model.Y model.Z]', model.TRIV', 1:numel(model.X), num_bins, radius, min_neighs)';


## Pre-trained models
* Currently we only provide a model trained for a small number of iterations (~1200) on the registered faust models. We will do our best to update this. Note that these are not the parameters used to produce the results published in the paper.  

## TODO
* Updtae pre-trained models
* Add postprocessing as done in the paper

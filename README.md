# Deep Functional Maps
This page contains a TensorFlow implementation (version 1.3.0) of the method described in https://arxiv.org/pdf/1704.08686

![Alt text](/fmnet.png?raw=true "Teaser")


## Instructions
* Download Data and Results folders from <todo: add google drive link>
* To run on a pair of shapes, update their filenames in test_pairs.txt
* run test_FMnet.py

## Data description
The network recieves as input a pair of shapes, in the format of a mat struct with precomputed shot descriptors and Laplacian eigenfunctions. See the example shapes provided in './Data/'.

## Pre-trained models
* Currently we only provide a model trained for a small number of iterations (~1200) on the registered faust models. We will do our best to update this. Note that these are not the parameters used to produce the results published in the paper.  

## TODO
* Updtae pre-trained models

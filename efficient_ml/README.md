# Quantization Demo

This repository contains simple notebooks to demonstrate post-training quantization using [qkeras](https://github.com/google/qkeras) and [TensoFlow modeloptimization](https://www.tensorflow.org/model_optimization/api_docs/python/tfmot/sparsity/keras/prune_low_magnitude).

------
## JupyterHub Instructions

This session will be done in a seperate jupyterhub instance.

### JupyterHub access (for hls4ml tutorial)
- Join hls4ml-tutorial GitHub Organization (check your email for invite) 
- Your should be able to see yourself here: [https://github.com/orgs/hls4ml-tutorial/people](https://github.com/orgs/hls4ml-tutorial/people)

### JupyterHub link
- Open [https://tutorials.fastmachinelearning.org](https://tutorials.fastmachinelearning.org) in your web browser
- Authenticate with your GitHub account (login if necessary)

-----

## Demo on Google Colaboratory
The quantization part of the demo could be done using Google Colaboratory (colab). No local setup is required in that case.

Google colab link: https://colab.research.google.com/drive/1EPD30XehpS8LC7mTk2PLi-T1F8jxaDpk?usp=sharing


## Setup the environment for running on local machines

### Step 1: Install Miniconda or Anaconda
Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html). You can find the installation instruction [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

### Step 2: Create the python environment

The Python environment used for the tutorials is specified in the `environment.yml` file. It can be setup like:
```bash
conda env create -f environment.yml
conda activate quant-demo
```

## Notebooks
The jet-tagging problem description could be found on the `hls4ml-tutorial` [slides](https://docs.google.com/presentation/d/1c4LvEc6yMByx2HJs8zUP5oxLtY6ACSizQdKvw5cg5Ck/edit#slide=id.ge9460ec16d_2_506) (page 30-32).

There are two notebooks:
- [part1_karas_training.ipynb](https://github.com/ml4fp/2024-lbnl/blob/main/efficient_ml/part1_karas_training.ipynb): This is the training notebook. It uses the open jet-tagging data and trains a Deep Neural Network (DNN) using Keras and TensorFlow.
- [part2_post-training-quantization.ipynb](https://github.com/ml4fp/2024-lbnl/blob/main/efficient_ml/part2_post-training-quantization.ipynb): This notebook shows how to do post-training quantization using qkeras. QKeras is "Quantized Keras" for deep heterogeneous quantization of ML models.
- [part3_pruning.ipynb] (https://github.com/ml4fp/2024-lbnl/blob/main/efficient_ml/part3_pruning.ipynb) This notebook shows how to do a apply magnitude-based pruning using Keras and TensorFlow.


## hls4ml-demo

We will use the official `hls4ml` tutorial repo

[https://github.com/fastmachinelearning/hls4ml-tutorial](https://github.com/fastmachinelearning/hls4ml-tutorial)


## Other resources:
- The official [hls4ml tutorial](https://github.com/fastmachinelearning/hls4ml-tutorial)
- QKeras [tutorial notebooks](https://github.com/google/qkeras/tree/master/notebook)

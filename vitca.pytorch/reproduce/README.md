# Attention-based Neural Cellular Automata

Welcome to the supplementalary section of our work. Here we provide a truncated version of our code (with unnecessary code removed) and a truncated version of our qualitative results (full version is over 3.7 GB). The Appendix is appended to the main manuscript. It is recommended to use the code provided at Mattie Tesfaldet's [Github](https://github.com/tesfaldet), which is also where full instructions on reproducing results will be provided.

Two folders are provided in the supplementary: `code` and `results`.

# Code

The `code` folder contains folders `masked_autoencoding` and `image_classification` for denoising autoencoding and linear probing, respectively. It also contains a `environment.yml` file which contains code dependencies (under an Anaconda environment).


## Installing dependencies
This project relies on Anaconda, PyTorch 1.10, an NVIDIA GPU with at least 48 GB of VRAM, and all dependencies listed in `environment.yml`. To install the conda environment with dependencies, execute the line below:

`conda env create -n pytorch1.10 --file environment.yml`

## Instructions for running experiments
This project relies on the Hydra framework to execute experiments. To train ViTCA on CelebA, for example, you would execute:

`python masked_autoencoding/train.py +experiment=[celeba_small] model=vitca`

The yaml files under the `conf/experiment` folder dictates the types of experiments that can be run. The yaml files under the `conf/model` folder dictates the models that can be used.

# Results

The `results` folder contains qualitative results using ViTCA and UNetCA (`CA results` folder) and qualitative results using UNet and ViT (`non-CA results` folder).

Results within the `CA results` folder are divided amongst the different experiments mentioned in the main manuscript and Appendix and are similarly named. The results are further divided into folders describing the noise configuration used (e.g., attn head masking/vitca/landcoverrep/2x2/50%) and in the case of attn head masking, the numbers indicate the index of the head masked (from top to bottom). For the hidden state viz results on MNIST, there is a special `convergence_animation` folder that shows an animation of the PCA manifold representing cells extracted at various ViTCA iterations from start to convergence. Videos will contain Fourier phase and magnitude visualizations for debugging purposes as well as hidden state visualizations using PCA such that spatial dimensions are preserved and the cell dim is reduced from 32 to 3. Since hidden values are between -1 and 1, there are positive and negative hidden PCA visualizations. This visualization is for debugging purposes only and is meant to be ignored.

Results within the `non-CA results` folder are divided in a self-explanatory fashion.
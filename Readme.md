# SpaGT

SpaGT (Spatially informed Graph Transformer)  is a python tool for spatial transcriptomics analysis.

![SpaGT](./SpaGT_overview.jpg)

# Abstract

Here, we present a novel graph transformer framework, SpaGT (Spatially informed Graph Transformer), which employs both expression and relational channels to model spatially aware graph representations. This approach enhances the denoising of gene expression data and facilitates the precise identification of spatial domains. Unlike traditional static localized convolutional aggregation, SpaGT leverages a structure-reinforced self-attention mechanism that iteratively evolves the structural information and transcriptional signal representation. Notably, SpaGT replaces graph convolution with global self-attention, thereby enabling the effective integration of global and spatially localized information, which enhances the detection of fine-grained spatial domains.

# Getting started

See [Documentation and Tutorials](https://spagt-tutorial.readthedocs.io/en/latest/index.html).

# Software dependencies

The dependencies for the codes are listed in requirements.txt

python==3.8

anndata==0.9.2

scanpy==1.9.8

numpy==1.22.0

pandas==2.0.3

scikit-learn==1.3.2

rpy2==3.5.15

torch==2.0.1


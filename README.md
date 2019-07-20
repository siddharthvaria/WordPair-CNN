# WordPair-CNN
Code repository for discourse relation prediction using word pair CNNs

common directory contains the scripts to prepare datasets from PDTB files (files ending in .pdtb)
src/theano_impl directory contains two classifiers:
- cnn_pdtb_arg_multiclass_jl.py
- cnn_pdtb_arg_wp_multiclass_jl.py

First one corresponds to extracting features from just arguments whereas the second one corresponds to extracting features from both arguments and their cartesian product.

Similarly, src/pytorch_impl directory contains files with same names.

To run the code, follow the below steps:

- parse pdtb files using common/parse_pdtb.py
- encode parsed pdtb data using common/encode_data.py script
- train/test the relevant classifiers using the encoded dataset.


### software requirements
- spacy
- ftfy
- scikit-learn
- Theano
- Lasagne
- pytorch
- gensim

Code is tested on Theano==0.9.0, Lasagne==0.2.dev1 and torch==0.4.1 (i.e pytorch)

Most of the scripts should work in python 3.6 except the theano implementations which are in python 2.7

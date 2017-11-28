# 5MT003
Project in molecular techniques in life science at the Karolinska Institute. Part of the Master's course in Molecular Techniques in Life Science associated with the SciLifeLab and conducted at the Royal Institute of Technology (KTH), Karolinska Institute &amp; Stockholm University.

# Project plan

## Improving compute efficiency of convolution based DNA function predictor

Elinor Löverli and Revant Gupta

Supervisor: Mikael Huss, SciLifeLab

### Project description: 

Convolutional layers are now being used to capture information from sequences like DNA however they are computationally expensive to train and prohibitively slow for larger datasets.

For our project we are planning to:

Benchmark the following method against non graphical ML methods like random forest.

Propose an alternate architecture that aims to provide faster model training.

### DanQ: a hybrid convolutional and recurrent deep neural network for quantifying the function of DNA sequences [github](https://github.com/uci-cbcl/DanQ) [Paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4914104/)

Made for predicting the function of non-protein coding DNA sequence. Uses a convolution layer to capture regulatory motifs (i e single DNA snippets that control the expression of genes, for instance), and a recurrent layer (of the LSTM type) to try to discover a “grammar” for how these single motifs work together. Based on Keras/Theano.

DanQ is a hybrid convolutional and recurrent neural network model for predicting the function of DNA *de novo* from sequence. 

### Citing DanQ:

Quang, D. and Xie, X. ''DanQ: a hybrid convolutional and recurrent neural network for predicting the function of DNA sequences'', NAR, 2015.

### Required

* Anconda python
* tensorflow
* keras 2.1.1

### Data:

You need to first download the training, validation, and testing sets from DeepSEA. You can download the datasets from [here](http://deepsea.princeton.edu/media/code/deepsea_train_bundle.v0.9.tar.gz). After you have extracted the contents of the tar.gz file, move the 3 .mat files into the data/ folder. 

# MACHINE LEARNING TOOLS

The ML subpackage is designed to explore the application of machine learning to filament and galaxy data, providing a framework to work with autoencoders and neural networks for dimensionality reduction and prediction tasks. This module introduces the basics of encoding galaxy and filament characteristics into a lower-dimensional latent space and then utilizing that latent space for predictive modeling, with the flexibility to make predictions about either galaxy properties, filament characteristics, or both. Developed by Daniel Gallego. 

This subpackage includes:

1. Core Functions:
   - ml_functions.py: Contains functions and core methods for implementing autoencoders and neural networks, including routines for data preprocessing, model training, and evaluation. These functions serve as the foundation for building and training machine learning models that can reduce data dimensionality and perform predictive tasks. 

2. Tutorials:
   - Neural_Networks.ipynb: This notebook provides a guided example of building and training an autoencoder and a neural network. It demonstrates encoding galaxy and filament data into a lower-dimensional latent space and then feeding this latent space representation into a neural network to predict various characteristics. Key sections include:
      - Setting up and training an autoencoder for dimensionality reduction.
      - Using the encoded latent space as input to a neural network.
      -  Making predictions about filament characteristics from galaxy properties (or vice versa) based on the latent space. 
    
    
Diamond Price Prediction with PyTorch:
This project is a Neural Network built from scratch to predict diamond prices based on the Seaborn Diamonds dataset.
I built this to practice Deep Learning, and to learn about Entity Embeddings and use it for categorical variables

Model Architecture:
The model is a Feed-Forward Network (MLP) built in PyTorch
Inputs: the embedding vectors of the categorical features, and the numerical features.
Hidden Layers: multiple layers layers of Linear -> ReLU -> BatchNorm -> Dropout.
Sizes: 128 -> 64 -> 32 -> 1 (Output).

How to run: python basic_NN_project.py

Libs:pandas numpy scikit-learn torch

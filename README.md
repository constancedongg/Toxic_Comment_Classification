# Toxic_Comment_Classification

## Goal
Build classifier to detect 6 different types of toxicity of comments: toxic, severe-toxic, insult, obscene, threat, identity_hate.


## Two Part 
1. Simplified version: binary classification of whether the comment is toxic; 
model: Logistic Regression, and SVM. 

2. Multi-categorical classification, model: Logistic regression and Keras LSTM.

## Results:
1. Binary classification: False positive: 0.08; False negative 0.12

2. Multi-categorical: Logistic: 0.968 CV score; Keras: 0.983

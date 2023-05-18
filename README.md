#Traffic Speed Prediction using GCN+LSTM

This repository contains the source code and necessary resources to implement a traffic speed prediction model using a combination of Graph Convolutional Networks (GCN) and Long Short-Term Memory (LSTM) networks.

##Problem Description

Traffic speed prediction is a significant challenge in the field of traffic management and planning. Accurate predictions of traffic speed can enable better traffic control strategies, route planning, and overall optimization of transportation systems. This repository aims to address this problem by providing a GCN+LSTM model that leverages the temporal and spatial dependencies in traffic data to forecast future traffic speeds.

##Model Architecture

The proposed model combines the power of GCN and LSTM to capture both the graph structure and temporal patterns in traffic data. The GCN layer processes the input traffic data, which is represented as a graph, to capture spatial dependencies between road segments. The output of the GCN layer is then fed into the LSTM layer, which learns temporal patterns in the data. The model is trained using historical traffic speed data and can be used to make predictions for future time steps.

##Repository Structure

data/: This directory contains the necessary dataset for training and evaluating the model. It may include files such as historical traffic speed data, road network information, and other relevant data.
scripts/: This directory contains the implementation of the GCN+LSTM model architecture, including the code for the GCN layer, LSTM layer, and any additional components.
train.py: This script is used to train the GCN+LSTM model using the provided dataset. It includes functions for data preprocessing, model training, and evaluation.
predict.py: This script allows the user to make predictions using the trained model. It takes as input the current traffic data and outputs the predicted traffic speeds for future time steps.
requirements.txt: This file lists all the required dependencies and their versions to run the code successfully.

##Getting Started

To use the GCN+LSTM model for traffic speed prediction, follow these steps:

Clone this repository: git clone https://github.com/alejandropolo/GCN_LSTM.git
Install the required dependencies: pip install -r requirements.txt
Prepare the dataset by placing the necessary files in the data/ directory.
Run the training script to train the model: python train.py
After training, you can use the predict.py script to make predictions.
Feel free to explore the code and customize it according to your specific requirements.

Conclusion
This repository provides a GCN+LSTM model for traffic speed prediction, which combines the strengths of graph convolutional networks and long short-term memory networks. By leveraging both spatial and temporal information in traffic data, this model can make accurate predictions of future traffic speeds. We hope this repository serves as a useful resource for researchers and practitioners working in the field of traffic management and transportation systems optimization.

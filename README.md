# LSTM_Model_selection
LSTM model selection

This repository contains a PyTorch-based implementation for automated model selection in Long Short-Term Memory (LSTM) neural networks. The notebook LSTM_Model_Selection.ipynb provides a structured framework for selecting both the number of hidden units and the optimal subset of input features based on Bayesian Information Criterion (BIC). The code defines a simple LSTM model (LSTMNet) and implements a series of algorithms for model selection. 
Algorithm 1 fits candidate LSTM models with multiple random initializations to mitigate local minima and computes the BIC score to assess model quality. 
Algorithm 2 performs hidden-node selection by iterating over candidate hidden sizes and choosing the one that minimizes BIC. 
Algorithm 3 implements input-node selection, which uses a greedy backward elimination approach to drop uninformative input nodes if doing so improves the BIC. Finally, we combines these steps into a full model selection procedure that alternates between tuning the hidden-node size and pruning input features until convergence. Users can specify the maximum hidden-node count (qmax) and the number of initializations (ninit). This framework can be adapted for time series regression, forecasting, or feature selection tasks involving sequential data.

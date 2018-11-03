CS 7641 Fall 2018 Assignment 3
This file describes the structure of this assignment submission. 
The assignment code is written in Python 3.5.1. Library dependencies are: 
scikit-learn 0.18.1
numpy 0.11.1
pandas 0.19.2
matplotlib 1.5.3
Other libraries used are part of the Python standard library.

The main folder has the following subfolders:

1. ./datasets folder with .the nested folders madelon/*, wine/* -> These are the original datasets, as downloaded from the UCI Machine Learning Repository http://archive.ics.uci.edu/ml/. At the time of writing, the site is down, reason why they're included in this repository.

2. ./preprocessed folder with madelon.hdf, wine.hdf -> A pre-processed/cleaned up copy of the datasets. These files are created by ./scripts/prepare_*.py

3. ./scripts folder contains
a. prepare_*.py -> These python scripts pre-process the original UCI ML repo files into a cleaner form for the experiments
b. helpers.py -> A collection of helper functions used for this assignment
c. dt_[madelon|wine].py" -> Code for the Decision Tree experiments
d. nn_[madelon|wine].py -> Code for the Neural Network Experiments
e. b_[madelon|wine].py -> Code for the Boosted Tree experiments
f. svm_[lin|rbf]_[madelon|wine].py -> Code for the Support Vector Machine (SVM) experiments with each of the kernels used
g. knn_[madelon|wine].py -> Code for the K-nearest Neighbors experiments

4. ./output. This folder contains the experimental results. 
Here, DT/ANN/Boost/KNN/SVM_Lin/SVM_RBF refer to decision trees, artificial neural networks, boosted trees, K-nearest neighbors, linear and RBF kernel SVMs respectively. A suffix of _OF indicates a deliberately "overfitted" version of the model where regularization is turned off.
The datasets are adult/madelon refering to the two datasets used (the UCI Adult dataset and the UCI Madelon dataset)
There are 75 files in this folder. They come the following types:
a. <Algorithm>_<dataset>_reg.csv -> The validation curve tests for <algorithm> on <dataset>
b. <Algorithn>_<dataset>_LC_train.scv -> Table of # of examples vs. CV training accuracy (for 5 folds) for <algorithm> on <dataset>. For learning curves
c. <Algorithn>_<dataset>_LC_test.csv -> Table of # of examples vs. CV testing accuracy (for 5 folds) for <algorithm> on <dataset>. For learning curves
d. <Algorithm>_<dataset>_timing.csv -> Table of fraction of training set vs. training and evaluation times. If the full training set is of size T and a fraction f are used for training, then the evaluation set is of size (T-fT)= (1-f)T
e. DT_<dataset>_nodecounts.csv -> Node counts for different pruning criteria in DT
f. <Algorithm>_<dataset>_tuned*.csv -> Tuned versions of interesting hyper parameter tuning problems.
g. "test results.csv" -> Table showing the optimal hyper-parameters chosen, as well as the final accuracy on the held out test set.

5. ./logs. This tracks the progress of each run following the pattern <Algorithm>_<Dataset>.out

Apart from the subfolders, the main folder has:

1. dguzman34-analysis.pdf -> The analysis for this assignment.
2. plot_[lc|ca].ipynb -> Jupiter notebooks to plot the learning and validation curves in the report
3. README.txt -> This file


To process the data run from a shell in the current folder:
./launch.sh

This will generate the output .csv files in ./output. To track the progress of each algorithm run, go to the ./logs folder and then execute less <Algorithm>_<Dataset>.out

To plot simply open a Jupiter notebook from the current folder
$jupyter notebook

Make sure the kernel corresponds to the python 3.5.1 implementation. Otherwise select Kernel ->  Change kernel to set it to the correct python configuration.
Click on Cell -> Run all


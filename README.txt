CS 7641 Fall 2018 Assignment 3
This file describes the structure of this assignment submission. 
The assignment code is written in Python 3.5.1. Library dependencies are: 
scikit-learn 0.18.1
numpy 0.11.1
pandas 0.19.2
matplotlib 1.5.3
Other libraries used are part of the Python standard library.

To process the data clone the following repository
https://github.gatech.edu/dguzman34/machine-learning-7641-a3.git

The main folder has the following subfolders:
BASE_<DATASET>
<Dataset> NN bmk.csv         Learning curve benchmark data for NN before clustering.

[PCA,ICA,RP,RF]_<DATASET>
<dataset> scree[1,2].csv     Principal component data. The Randomized Projection folder contains two separate runs.
<dataset> dim red.csv        Grid search curves after applying dimensionality reduction and NN consecutively.

[BASE,PCA,ICA,RP,RF]_<DATASET>
datasets.hdf                 Original dataset (for BASE) or Reduced Dimension versions of the dataset.
<Dataset>2D.csv              2D Visualization data of the dataset.
SSE.csv                      K-means - Sum of Squared Errors.
<Dataset> logliklihood.csv   EM - Log Likelihood results.
<Dataset> adjMI.csv          K-means and EM - Adjusted Mutual Information Scores.
<Dataset> acc.csv            K-means and EM - Accuracy between dataset labels and cluster labels.
<Dataset> cluster Kmeans.csv Grid search curves after applying Kmeans and NN consecutively.
<Dataset> cluster GMM.csv    Grid search curves after applying EM and NN consecutively.

Run from a shell in the current folder:
./run.sh

This will generate the output .csv files in each folder.

To plot simply go to the /plots folder and open a Jupiter notebook from the current folder
$jupyter notebook

Make sure the kernel corresponds to the python 3.5.1 implementation. Otherwise select Kernel ->  Change kernel to set it to the correct python configuration.
Click on Cell -> Run all


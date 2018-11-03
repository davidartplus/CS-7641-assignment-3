
BASE_<DATASET>
<Dataset> NN bmk.csv         Learning curve benchmark data for NN before clustering.

[PCA,ICA,RP,RF]_<DATASET>
<dataset> scree[1,2].csv     Principal component data. The Randomized Projection folder contains two separate runs.
<dataset> dim red.csv        Grid search curves after applying PCA and NN consecutively.

[BASE,PCA,ICA,RP,RF]_<DATASET>
datasets.hdf                 Original dataset (for BASE) or Reduced Dimension versions of the dataset.
<Dataset>2D.csv              2D Visualization data of the dataset.
SSE.csv                      K-means - Sum of Squared Errors.
<Dataset> logliklihood.csv   EM - Log Likelihood results.
<Dataset> adjMI.csv          K-means and EM - Adjusted Mutual Information Scores.
<Dataset> acc.csv            K-means and EM - Accuracy between dataset labels and cluster labels.
<Dataset> cluster Kmeans.csv Grid search curves after applying Kmeans and NN consecutively. 
<Dataset> cluster GMM.csv    Grid search curves after applying EM and NN consecutively.


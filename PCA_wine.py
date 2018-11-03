import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

out = './PCA_WINE/'
cmap = cm.get_cmap('Spectral')

np.random.seed(0)

wine = pd.read_hdf('./BASE_WINE/datasets.hdf', 'wine')
wineX = wine.drop('quality', 1).copy().values
wineY = wine['quality'].copy().values

wineX = StandardScaler().fit_transform(wineX)

clusters = [2, 5, 10, 15, 20, 25, 30, 35, 40]
dims = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

nn_arch=[(7,), (15,), (31,), (62,), (124,)]
nn_reg=[10.0, 0.01, 1e-05, 1e-08]
# raise
# %% data for 1

pca = PCA(random_state=5)
pca.fit(wineX)
# Out of the 13 attributes, 11 have the largest variance
tmp = pd.Series(data=pca.explained_variance_, index=range(1, 12))
tmp.to_csv(out + 'wine scree.csv')

# %% Data for 2

grid = {'pca__n_components': dims, 'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': nn_arch}
pca = PCA(random_state=5)
mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5)
pipe = Pipeline([('pca', pca), ('NN', mlp)])
gs = GridSearchCV(pipe, grid, verbose=10, cv=5) #n_jobs=-1 makes it slower???

gs.fit(wineX, wineY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out + 'Wine dim red.csv')

# %% data for 3
# Set this from chart 2 and dump, use clustering script to finish up
dim = 5
pca = PCA(n_components=dim, random_state=10)
wineX2 = pca.fit_transform(wineX)
wine2 = pd.DataFrame(np.hstack((wineX2, np.atleast_2d(wineY).T)))
cols = list(range(wine2.shape[1]))
cols[-1] = 'quality'
wine2.columns = cols
wine2.to_hdf(out + 'datasets.hdf', 'wine', complib='blosc', complevel=9)

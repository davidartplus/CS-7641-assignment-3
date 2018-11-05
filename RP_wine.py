# %% Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from collections import defaultdict
from helpers import pairwiseDistCorr, reconstructionError
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from itertools import product

out = './RP_WINE/'
cmap = cm.get_cmap('Spectral')

np.random.seed(0)

wine = pd.read_hdf('./BASE_WINE/datasets.hdf', 'wine')
wineX = wine.drop('quality', 1).copy().values
wineY = wine['quality'].copy().values

wineX = StandardScaler().fit_transform(wineX)

clusters = [2, 5, 10, 15]
dims = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

nn_arch=[(7,), (15,), (31,), (62,), (124,)]
nn_reg=[10.0, 0.01, 1e-05, 1e-08]

# raise
# %% data for 1

tmp = defaultdict(dict)
for i, dim in product(range(10), dims):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    tmp[dim][i] = pairwiseDistCorr(rp.fit_transform(wineX), wineX)
tmp = pd.DataFrame(tmp).T
tmp.to_csv(out + 'wine scree1.csv')

tmp = defaultdict(dict)
for i, dim in product(range(10), dims):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    rp.fit(wineX)
    tmp[dim][i] = reconstructionError(rp, wineX)
tmp = pd.DataFrame(tmp).T
tmp.to_csv(out + 'wine scree2.csv')

# %% Data for 2

grid = {'rp__n_components': dims, 'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': nn_arch}
rp = SparseRandomProjection(random_state=5)
mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5)
pipe = Pipeline([('rp', rp), ('NN', mlp)])
gs = GridSearchCV(pipe, grid, verbose=10, cv=5)

gs.fit(wineX, wineY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out + 'Wine dim red.csv')

# %% data for 3
# Set this from chart 2 and dump, use clustering script to finish up
dim = 10
rp = SparseRandomProjection(n_components=dim, random_state=5)

wineX2 = rp.fit_transform(wineX)
wine2 = pd.DataFrame(np.hstack((wineX2, np.atleast_2d(wineY).T)))
cols = list(range(wine2.shape[1]))
cols[-1] = 'quality'
wine2.columns = cols
wine2.to_hdf(out + 'datasets.hdf', 'wine', complib='blosc', complevel=9)

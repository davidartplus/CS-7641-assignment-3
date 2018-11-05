# %% Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from collections import defaultdict
from helpers import pairwiseDistCorr, nn_reg, nn_arch, reconstructionError
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from itertools import product

out = './RP_MADELON/'
cmap = cm.get_cmap('Spectral')

np.random.seed(0)

madelon = pd.read_hdf('./BASE_MADELON/datasets.hdf', 'madelon')
madelonX = madelon.drop('Class', 1).copy().values
madelonY = madelon['Class'].copy().values

madelonX = StandardScaler().fit_transform(madelonX)

clusters = [2, 5, 10, 15, 20, 25, 30, 35, 40]
dims = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
# raise
# %% data for 1

tmp = defaultdict(dict)
for i, dim in product(range(10), dims):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    tmp[dim][i] = pairwiseDistCorr(rp.fit_transform(madelonX), madelonX)
tmp = pd.DataFrame(tmp).T
tmp.to_csv(out + 'madelon scree1.csv')

tmp = defaultdict(dict)
for i, dim in product(range(10), dims):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    rp.fit(madelonX)
    tmp[dim][i] = reconstructionError(rp, madelonX)
tmp = pd.DataFrame(tmp).T
tmp.to_csv(out + 'madelon scree2.csv')

# %% Data for 2

grid = {'rp__n_components': dims, 'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': nn_arch}
rp = SparseRandomProjection(random_state=5)
mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5)
pipe = Pipeline([('rp', rp), ('NN', mlp)])
gs = GridSearchCV(pipe, grid, verbose=10, cv=5)

gs.fit(madelonX, madelonY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out + 'Madelon dim red.csv')

# %% data for 3
# Set this from chart 2 and dump, use clustering script to finish up
dim = 60
rp = SparseRandomProjection(n_components=dim, random_state=5)

madelonX2 = rp.fit_transform(madelonX)
madelon2 = pd.DataFrame(np.hstack((madelonX2, np.atleast_2d(madelonY).T)))
cols = list(range(madelon2.shape[1]))
cols[-1] = 'Class'
madelon2.columns = cols
madelon2.to_hdf(out + 'datasets.hdf', 'madelon', complib='blosc', complevel=9)

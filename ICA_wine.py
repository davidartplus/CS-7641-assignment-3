# %% Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import FastICA

out = './ICA_WINE/'

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

ica = FastICA(random_state=5)
kurt = {}
for dim in dims:
    ica.set_params(n_components=dim)
    tmp = ica.fit_transform(wineX)
    tmp = pd.DataFrame(tmp)
    tmp = tmp.kurt(axis=0)
    kurt[dim] = tmp.abs().mean()

kurt = pd.Series(kurt)
kurt.to_csv(out + 'wine scree.csv')

# %% Data for 2

grid = {'ica__n_components': dims, 'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': nn_arch}
ica = FastICA(random_state=5)
mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5)
pipe = Pipeline([('ica', ica), ('NN', mlp)])
gs = GridSearchCV(pipe, grid, verbose=10, cv=5)

gs.fit(wineX, wineY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out + 'Wine dim red.csv')

# %% data for 3
# Set this from chart 2 and dump, use clustering script to finish up
dim = 45
ica = FastICA(n_components=dim, random_state=10)

wineX2 = ica.fit_transform(wineX)
wine2 = pd.DataFrame(np.hstack((wineX2, np.atleast_2d(wineY).T)))
cols = list(range(wine2.shape[1]))
cols[-1] = 'quality'
wine2.columns = cols
wine2.to_hdf(out + 'datasets.hdf', 'wine', complib='blosc', complevel=9)

# %% Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import nn_arch, nn_reg
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import FastICA

out = './ICA_MADELON/'

np.random.seed(0)

madelon = pd.read_hdf('./BASE_MADELON/datasets.hdf', 'madelon')
madelonX = madelon.drop('Class', 1).copy().values
madelonY = madelon['Class'].copy().values

madelonX = StandardScaler().fit_transform(madelonX)

clusters = [2, 5, 10, 15, 20, 25, 30, 35, 40]
dims = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
# raise
# %% data for 1

ica = FastICA(random_state=5)
kurt = {}
for dim in dims:
    ica.set_params(n_components=dim)
    tmp = ica.fit_transform(madelonX)
    tmp = pd.DataFrame(tmp)
    tmp = tmp.kurt(axis=0)
    kurt[dim] = tmp.abs().mean()

kurt = pd.Series(kurt)
kurt.to_csv(out + 'madelon scree.csv')

# %% Data for 2

grid = {'ica__n_components': dims, 'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': nn_arch}
ica = FastICA(random_state=5)
mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5)
pipe = Pipeline([('ica', ica), ('NN', mlp)])
gs = GridSearchCV(pipe, grid, verbose=10, cv=5)

gs.fit(madelonX, madelonY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out + 'Madelon dim red.csv')

# %% data for 3
# Set this from chart 2 and dump, use clustering script to finish up
dim = 45
ica = FastICA(n_components=dim, random_state=10)

madelonX2 = ica.fit_transform(madelonX)
madelon2 = pd.DataFrame(np.hstack((madelonX2, np.atleast_2d(madelonY).T)))
cols = list(range(madelon2.shape[1]))
cols[-1] = 'Class'
madelon2.columns = cols
madelon2.to_hdf(out + 'datasets.hdf', 'madelon', complib='blosc', complevel=9)

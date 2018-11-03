# %% Imports
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import ImportanceSelect
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

nn_arch=[(7,), (15,), (31,), (62,), (124,)]
nn_reg=[10.0, 0.01, 1e-05, 1e-08]

if __name__ == '__main__':
    out = './RF_WINE/'

    np.random.seed(0)

    wine = pd.read_hdf('./BASE_WINE/datasets.hdf', 'wine')
    wineX = wine.drop('quality', 1).copy().values
    wineY = wine['quality'].copy().values

    wineX = StandardScaler().fit_transform(wineX)

    clusters = [2, 5, 10, 15, 20, 25, 30, 35, 40]
    dims = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    # %% data for 1

    rfc = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=5, n_jobs=7)
    fs_wine = rfc.fit(wineX, wineY).feature_importances_

    tmp = pd.Series(np.sort(fs_wine)[::-1])
    tmp.to_csv(out + 'wine scree.csv')

    # %% Data for 2
    filtr = ImportanceSelect(rfc)
    grid = {'filter__n': dims, 'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': nn_arch}
    mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5)
    pipe = Pipeline([('filter', filtr), ('NN', mlp)])
    gs = GridSearchCV(pipe, grid, verbose=10, cv=5)

    gs.fit(wineX, wineY)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out + 'Wine dim red.csv')

    grid = {'filter__n': dims, 'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': nn_arch}
    mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5)
    pipe = Pipeline([('filter', filtr), ('NN', mlp)])
    gs = GridSearchCV(pipe, grid, verbose=10, cv=5)

    # %% data for 3
    # Set this from chart 2 and dump, use clustering script to finish up
    dim = 20
    filtr = ImportanceSelect(rfc, dim)

    wineX2 = filtr.fit_transform(wineX, wineY)
    wine2 = pd.DataFrame(np.hstack((wineX2, np.atleast_2d(wineY).T)))
    cols = list(range(wine2.shape[1]))
    cols[-1] = 'quality'
    wine2.columns = cols
    wine2.to_hdf(out + 'datasets.hdf', 'wine', complib='blosc', complevel=9)

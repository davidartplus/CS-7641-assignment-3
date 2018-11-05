# %% Imports
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import nn_arch, nn_reg, ImportanceSelect
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
    out = './RF_MADELON/'

    np.random.seed(0)

    madelon = pd.read_hdf('./BASE_MADELON/datasets.hdf', 'madelon')
    madelonX = madelon.drop('Class', 1).copy().values
    madelonY = madelon['Class'].copy().values

    madelonX = StandardScaler().fit_transform(madelonX)

    clusters = [2, 5, 10, 15, 20, 25, 30, 35, 40]
    dims = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

    # %% data for 1

    rfc = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=5, n_jobs=7)
    fs_madelon = rfc.fit(madelonX, madelonY).feature_importances_

    tmp = pd.Series(np.sort(fs_madelon)[::-1])
    tmp.to_csv(out + 'madelon scree.csv')

    # %% Data for 2
    filtr = ImportanceSelect(rfc)
    grid = {'filter__n': dims, 'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': nn_arch}
    mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5)
    pipe = Pipeline([('filter', filtr), ('NN', mlp)])
    gs = GridSearchCV(pipe, grid, verbose=10, cv=5)

    gs.fit(madelonX, madelonY)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out + 'Madelon dim red.csv')

    grid = {'filter__n': dims, 'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': nn_arch}
    mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5)
    pipe = Pipeline([('filter', filtr), ('NN', mlp)])
    gs = GridSearchCV(pipe, grid, verbose=10, cv=5)

    # %% data for 3
    # Set this from chart 2 and dump, use clustering script to finish up
    dim = 15
    filtr = ImportanceSelect(rfc, dim)

    madelonX2 = filtr.fit_transform(madelonX, madelonY)
    madelon2 = pd.DataFrame(np.hstack((madelonX2, np.atleast_2d(madelonY).T)))
    cols = list(range(madelon2.shape[1]))
    cols[-1] = 'Class'
    madelon2.columns = cols
    madelon2.to_hdf(out + 'datasets.hdf', 'madelon', complib='blosc', complevel=9)

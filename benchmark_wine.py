import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


out = './BASE_WINE/'
np.random.seed(0)

wine = pd.read_hdf('./BASE_WINE/datasets.hdf', 'wine')
wineX = wine.drop('quality', 1).copy().values
wineY = wine['quality'].copy().values

wineX = StandardScaler().fit_transform(wineX)

nn_arch=[(7,), (15,), (31,), (62,), (124,)]
nn_reg=[10.0, 0.01, 1e-05, 1e-08]

# %% benchmarking for chart type 2

grid = {'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': nn_arch}
mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5)
pipe = Pipeline([('NN', mlp)])
gs = GridSearchCV(pipe, grid, verbose=10, cv=5, n_jobs=-1)

gs.fit(wineX, wineY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out + 'Wine NN bmk.csv')

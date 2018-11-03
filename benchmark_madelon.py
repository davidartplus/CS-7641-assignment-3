import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
# from helpers import nn_reg, nn_arch

nn_reg=[10.0, 0.01, 1e-05, 1e-08]
nn_arch=[(7,), (15,), (31,), (62,), (124,)]

out = './BASE_MADELON/'
np.random.seed(0)

madelon = pd.read_hdf('./BASE_MADELON/datasets.hdf', 'madelon')
madelonX = madelon.drop('Class', 1).copy().values
madelonY = madelon['Class'].copy().values

madelonX = StandardScaler().fit_transform(madelonX)

# %% benchmarking for chart type 2

grid = {'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': nn_arch}
mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5)
pipe = Pipeline([('NN', mlp)])
gs = GridSearchCV(pipe, grid, verbose=10, cv=5, n_jobs=-1)

gs.fit(madelonX, madelonY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out + 'Madelon NN bmk.csv')

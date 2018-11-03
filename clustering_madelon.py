import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from time import clock
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans as kmeans
from sklearn.mixture import GaussianMixture as GMM
from collections import defaultdict
from helpers import cluster_acc, myGMM, nn_arch, nn_reg
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import sys

out = './{}_MADELON/'.format(sys.argv[1])

madelon = pd.read_hdf(out + 'datasets.hdf', 'madelon')
madelonX = madelon.drop('Class', 1).copy().values
madelonY = madelon['Class'].copy().values

clusters = [2, 5, 20, 32, 38]

# %% Data for 1-3
SSE = defaultdict(dict)
ll = defaultdict(dict)
acc = defaultdict(lambda: defaultdict(dict))
adjMI = defaultdict(lambda: defaultdict(dict))
km = kmeans(random_state=5)
gmm = GMM(random_state=5)

st = clock()
for k in clusters:
    km.set_params(n_clusters=k)
    gmm.set_params(n_components=k)
    km.fit(madelonX)
    gmm.fit(madelonX)
    SSE[k]['Madelon'] = km.score(madelonX)
    ll[k]['Madelon'] = gmm.score(madelonX)
    acc[k]['Madelon']['Kmeans'] = cluster_acc(madelonY, km.predict(madelonX))
    acc[k]['Madelon']['GMM'] = cluster_acc(madelonY, gmm.predict(madelonX))
    adjMI[k]['Madelon']['Kmeans'] = ami(madelonY, km.predict(madelonX))
    adjMI[k]['Madelon']['GMM'] = ami(madelonY, gmm.predict(madelonX))

SSE = (-pd.DataFrame(SSE)).T
SSE.rename(columns=lambda x: x + ' SSE (left)', inplace=True)
ll = pd.DataFrame(ll).T
ll.rename(columns=lambda x: x + ' log-likelihood', inplace=True)
acc = pd.Panel(acc)
adjMI = pd.Panel(adjMI)

SSE.to_csv(out + 'SSE.csv')
ll.to_csv(out + 'logliklihood.csv')
acc.ix[:, :, 'Madelon'].to_csv(out + 'Madelon acc.csv')
adjMI.ix[:, :, 'Madelon'].to_csv(out + 'Madelon adjMI.csv')

# %% NN fit data (2,3)

grid = {'km__n_clusters': clusters, 'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': nn_arch}
mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5)
km = kmeans(random_state=5)
pipe = Pipeline([('km', km), ('NN', mlp)])
gs = GridSearchCV(pipe,  grid, verbose=10)

gs.fit(madelonX, madelonY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out + 'Madelon cluster Kmeans.csv')

grid = {'gmm__n_components': clusters, 'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': nn_arch}
mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5)
gmm = myGMM(random_state=5)
pipe = Pipeline([('gmm', gmm), ('NN', mlp)])
gs = GridSearchCV(pipe, grid, verbose=10, cv=5)

gs.fit(madelonX, madelonY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out + 'Madelon cluster GMM.csv')

# %% For chart 4/5
madelonX2D = TSNE(verbose=10, random_state=5).fit_transform(madelonX)

madelon2D = pd.DataFrame(np.hstack((madelonX2D, np.atleast_2d(madelonY).T)), columns=['x', 'y', 'target'])

madelon2D.to_csv(out + 'madelon2D.csv')

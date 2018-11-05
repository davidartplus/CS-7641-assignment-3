import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from time import clock
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans as kmeans
from sklearn.mixture import GaussianMixture as GMM
from collections import defaultdict
from helpers import cluster_acc, myGMM
# from helpers import nn_arch, nn_reg
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import sys

out = './{}_WINE/'.format(sys.argv[1])

wine = pd.read_hdf(out + 'datasets.hdf', 'wine')
wineX = wine.drop('quality', 1).copy().values
wineY = wine['quality'].copy().values

# clusters = [2]
clusters = [2, 5, 10, 15]
nn_arch=[(7,), (15,), (31,), (62,), (124,)]
nn_reg=[10.0, 0.01, 1e-05, 1e-08]

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
    km.fit(wineX)
    gmm.fit(wineX)
    SSE[k]['Wine'] = km.score(wineX)
    ll[k]['Wine'] = gmm.score(wineX)
    acc[k]['Wine']['Kmeans'] = cluster_acc(wineY, km.predict(wineX))
    acc[k]['Wine']['GMM'] = cluster_acc(wineY, gmm.predict(wineX))
    adjMI[k]['Wine']['Kmeans'] = ami(wineY, km.predict(wineX))
    adjMI[k]['Wine']['GMM'] = ami(wineY, gmm.predict(wineX))

SSE = (-pd.DataFrame(SSE)).T
SSE.rename(columns=lambda x: x + ' SSE (left)', inplace=True)
ll = pd.DataFrame(ll).T
ll.rename(columns=lambda x: x + ' log-likelihood', inplace=True)
acc = pd.Panel(acc)
adjMI = pd.Panel(adjMI)

SSE.to_csv(out + 'SSE.csv')
ll.to_csv(out + 'logliklihood.csv')
acc.ix[:, :, 'Wine'].to_csv(out + 'Wine acc.csv')
adjMI.ix[:, :, 'Wine'].to_csv(out + 'Wine adjMI.csv')

# %% NN fit data (2,3)

grid = {'km__n_clusters': clusters, 'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': nn_arch}
mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5)
km = kmeans(random_state=5)
pipe = Pipeline([('km', km), ('NN', mlp)])
gs = GridSearchCV(pipe,  grid, verbose=10)

gs.fit(wineX, wineY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out + 'Wine cluster Kmeans.csv')

grid = {'gmm__n_components': clusters, 'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': nn_arch}
mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5)
gmm = myGMM(random_state=5)
pipe = Pipeline([('gmm', gmm), ('NN', mlp)])
gs = GridSearchCV(pipe, grid, verbose=10, cv=5)

gs.fit(wineX, wineY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out + 'wine cluster GMM.csv')

# %% For chart 4/5
wineX2D = TSNE(verbose=10, random_state=5).fit_transform(wineX)

wine2D = pd.DataFrame(np.hstack((wineX2D, np.atleast_2d(wineY).T)), columns=['x', 'y', 'target'])

wine2D.to_csv(out + 'wine2D.csv')

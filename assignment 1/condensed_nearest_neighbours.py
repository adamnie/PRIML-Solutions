import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from itertools import product
import seaborn as sns
from sklearn.neighbors import DistanceMetric
from imblearn.under_sampling import CondensedNearestNeighbour

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

NUM_ROWS = 2
NUM_COLS = 4

cnn = CondensedNearestNeighbour()
X_resampled, y_resampled = cnn.fit_sample(X, y)

background_color = ListedColormap(sns.color_palette("bright", 3).as_hex())
data_point_color = ListedColormap(sns.color_palette("dark", 3).as_hex())

metrics = ['euclidean', 'mahalanobis']

n_neighbors = [1,3]
h = .02
num_neighbours = 5
datasets = [{"X": X, "y": y, "cnn": False}, {"X": X_resampled, "y": y_resampled, "cnn": True}]

f, axes = plt.subplots(nrows=NUM_ROWS, ncols=NUM_COLS)
f.tight_layout()

i = 0

for metric, n, data in product(metrics, n_neighbors, datasets):

    X = data["X"]
    y = data["y"]

    if metric == 'mahalanobis':
        metric_params = { "V": np.cov(X[:, 0], X[:, 1], rowvar=0) }
    else:
        metric_params = None

    clf = neighbors.KNeighborsClassifier(n, weights='uniform', metric=metric, metric_params=metric_params)
    clf.fit(X, y)

    x_min, x_max = X[:,0].min() - .5, X[:,0].max() + .5
    y_min, y_max = X[:,1].min() - .5, X[:,1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    plot = axes[i//NUM_COLS, i%NUM_COLS]
    i = i+1
    plot.pcolormesh(xx, yy, Z, cmap=background_color)

    plot.scatter(X[:,0], X[:,1],c=y, cmap=data_point_color)

    plot.set_xlim(xx.min(), xx.max())
    plot.set_ylim(yy.min(), yy.max())
    plot.set_xticks(())
    plot.set_yticks(())
    plot.set_title("k = %i, metric = '%s', cnn = '%s'" % (n, metric, data["cnn"]))

plt.show()

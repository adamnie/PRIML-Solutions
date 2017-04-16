from scipy import misc
from pyclustering.cluster import xmeans
from pyclustering.utils import draw_clusters
from matplotlib.mlab import PCA
from matplotlib.pyplot import plot as plt
import numpy as np


def remove_duplicates_pixels(image):
    pixel_list = image.reshape(-1, image.shape[-1])
    return np.vstack({tuple(pixel_rgb) for pixel_rgb in pixel_list})


def calculate_clusters_and_save_plot(data, plot_name, tolerance=0.025, kmax=20):
    centers = data[np.random.choice(data.shape[0], NUM_INIT_CLUSTERS, replace=False)]
    xmeans_instance = xmeans.xmeans(data, initial_centers=centers, tolerance=tolerance,
                                    criterion=xmeans.splitting_type.BAYESIAN_INFORMATION_CRITERION,
                                    kmax=kmax, ccore=False)
    xmeans_instance.process()
    clusters = xmeans_instance.get_clusters()
    centers = xmeans_instance.get_centers()

    plot = draw_clusters(unique_pixels, clusters)
    plot.get_figure().save_fig(plot_name, dpi=200)

    return clusters, centers

IMG_PATH = './img/COLORFUL2.jpg'
NUM_INIT_CLUSTERS = 2
image = misc.imread(IMG_PATH)

unique_pixels = remove_duplicates_pixels(image)
pixels = image.reshape(-1, image.shape[-1])

clusters_unique, centers_unique = calculate_clusters_and_save_plot(unique_pixels, './img/unique_plot.png')
clusters, centers = calculate_clusters_and_save_plot(pixels, './img/duplicates_plot.png')

pca_unique = PCA(unique_pixels).Y
pca_duplicates = PCA(pixels).Y


plt.figure();
plt.scatter(pca_unique[:, 0], pca_unique[:, 1], c=unique_pixels)
# plt.savefig()

plt.figure()
plt.scatter(pca_unique[:, 0], pca_unique[:, 1], c=unique_pixels)
# plt.savefig()
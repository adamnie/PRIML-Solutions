from scipy import misc
from pyclustering.cluster import xmeans
from pyclustering.utils import draw_clusters
import numpy as np


def remove_duplicates_pixels(image):
    pixel_list = image.reshape(-1, image.shape[-1])
    return np.vstack({tuple(pixel_rgb) for pixel_rgb in pixel_list})


IMG_PATH = './img/COLORFUL2.jpg'
NUM_INIT_CLUSTERS = 2
image = misc.imread(IMG_PATH)


unique_pixels = remove_duplicates_pixels(image)
centers = unique_pixels[np.random.choice(unique_pixels.shape[0], NUM_INIT_CLUSTERS, replace=False)] # random two data points
xmeans_instance = xmeans.xmeans(unique_pixels, initial_centers=centers, tolerance=0.025,
                                criterion=xmeans.splitting_type.BAYESIAN_INFORMATION_CRITERION, kmax=20, ccore=False)

xmeans_instance.process()
clusters = xmeans_instance.get_clusters()
centers = xmeans_instance.get_centers()


draw_clusters(unique_pixels, clusters)
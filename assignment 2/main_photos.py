import os
import pickle
import numpy as np
from scipy import misc
from sklearn.decomposition import PCA

from matplotlib import pyplot as plt

path = '/home/adam/machine_learning_photo_dataset/'
pickled_photos_filename = './photos.p'


def load_images_with_labels(root_path):
    images = []
    labels = []
    for subdir, dirs, files in os.walk(root_path):
        for file in files:
            filepath = subdir + os.sep + file
            #directories with photos with specific persons are named "1", "2", etc
            label = int(subdir.split(os.sep)[-1])
            if filepath.endswith('.jpg'): #I should add all relevant extensions
                image = misc.imread(filepath, flatten=True)
                images.append(image)
                labels.append(label)

    return images, labels


def shuffle_dataset(dataset):
    pass


def maybe_apply_PCA(dataset, force=False):
    if not os.path.isfile(pickled_photos_filename) or force:
        print('No saved data found. Performing PCA')
        pca_dataset = apply_PCA(dataset)
        pickle.dump(pca_dataset, open(pickled_photos_filename, 'wb'))

    else:
        print('Pickled data found. Opening previous data instead of performing PCA')
        pca_dataset = pickle.load(open(pickled_photos_filename, 'rb'))

    return pca_dataset

def apply_PCA(dataset):
    pca = PCA()
    dataset_pca = []
    for image in dataset:
        dataset_pca.append(pca.fit_transform(image))
    return dataset_pca

def reverse_PCA(dataset):
    pass

def plot_PCAed_images(pca_dataset):
    for image in pca_dataset:
        plt.figure()
        plt.imshow(image, cmap='gray')
    plt.show()

def plot_average_image(dataset):
    average_image = np.mean(np.array(dataset), axis=0)
    plt.figure()
    plt.imshow(average_image, cmap='gray')
    plt.show()

def reduce_dims_to(new_num_dims):
    pass

def plot_on_2D_surface(dataset):
    pass

photos,labels = load_images_with_labels(path)

pca_photos = maybe_apply_PCA(photos, force=False)
# plot_PCAed_images(pca_photos)

plot_average_image(photos)
plot_average_image(pca_photos)

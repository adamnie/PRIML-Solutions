from dataset_generator import DatasetGenerator
from dataset_generator import Point
from matplotlib.mlab import PCA

import numpy as np
import matplotlib.pyplot as plt

#constants
center = Point(10, 10)
r = 5
R = 10
slope_asc = 4
slope_desc = -4
x = 0
y = 1
NUM_EXMPL = 1000

generator = DatasetGenerator()

ring_dataset = generator.ring_dataset(center, r, R)
circle_dataset = generator.circle_dataset(center, r)
asc_dataset = generator.linear_dataset(center, slope_asc)
desc_dataset = generator.linear_dataset(center, slope_desc)

fig = plt.figure(1)
fig.suptitle('Ordinairy PCA')
sub1 = fig.add_subplot(221)
sub1.set_title('Circlular dataset no PCA')
sub1.plot(circle_dataset[:, x], circle_dataset[:, y], '+r')
sub1.plot(ring_dataset[:, x], ring_dataset[:, y], '+b')

sub2 = fig.add_subplot(222)
sub2.set_title('Linear dataset no PCA')
sub2.plot(asc_dataset[:, x], asc_dataset[:, y], '+r')
sub2.plot(desc_dataset[:, x], desc_dataset[:, y], '+b')

joined_circle = np.concatenate((circle_dataset, ring_dataset))
pca_circle = PCA(joined_circle).Y

joined_linear = np.concatenate((asc_dataset, desc_dataset))
pca_linear = PCA(joined_linear).Y

sub3 = fig.add_subplot(223)
sub3.set_title('Circular dataset PCA joined sets')
sub3.plot(pca_circle[:NUM_EXMPL, x], pca_circle[:NUM_EXMPL, y], '+r')
sub3.plot(pca_circle[NUM_EXMPL:, x], pca_circle[NUM_EXMPL:, y], '+b')

sub4 = fig.add_subplot(224)
sub4.set_title('Linear dataset PCA joined sets')
sub4.plot(pca_linear[:NUM_EXMPL, x], pca_linear[:NUM_EXMPL, y], '+r')
sub4.plot(pca_linear[NUM_EXMPL:, x], pca_linear[NUM_EXMPL:, y], '+b')

# kernel tricks
from sklearn.decomposition import KernelPCA



fig2 = plt.figure()
sub21 = fig2.add_subplot(221)
sub21.set_title('Circular dataset no PCA')
sub21.plot(circle_dataset[:, x], circle_dataset[:, y], '+r')
sub21.plot(ring_dataset[:, x], ring_dataset[:, y], '+b')

sub22 = fig2.add_subplot(222)
sub22.set_title('Linear dataset no PCA')
sub22.plot(asc_dataset[:, x], asc_dataset[:, y], '+r')
sub22.plot(desc_dataset[:, x], desc_dataset[:, y], '+b')

kpca = KernelPCA(kernel="rbf")

KPCA_circle = kpca.fit_transform(joined_circle)
KPCA_linear = kpca.fit_transform(joined_linear)

sub23 = fig2.add_subplot(223)
sub23.set_title('Circular dataset kernel PCA')
sub23.plot(KPCA_circle[NUM_EXMPL:, x], KPCA_circle[NUM_EXMPL:, y], '+r')
sub23.plot(KPCA_circle[:NUM_EXMPL, x], KPCA_circle[:NUM_EXMPL, y], '+b')

sub24 = fig2.add_subplot(224)
sub24.set_title('Linear dataset kernel PCA')
sub24.plot(KPCA_linear[NUM_EXMPL:, x], KPCA_linear[NUM_EXMPL:, y], '+r')
sub24.plot(KPCA_linear[:NUM_EXMPL, x], KPCA_linear[:NUM_EXMPL, y], '+b')

plt.show()
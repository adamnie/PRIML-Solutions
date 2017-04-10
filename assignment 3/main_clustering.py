from dataset_generator import Point, DatasetGenerator
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def scale_center_points(points, scaling_factor=10):
    for point in points:
        point.x = scaling_factor * point.x
        point.y = scaling_factor * point.y

    return points


def plot_data_set(data_set):
    plt.figure()
    plt.title('Generated 9 circular data clusters')
    plt.plot(data_set[:, 0], data_set[:, 1], 'r+')
    plt.show()


def get_silhouette_scores(data, init, max_iter=20, num_runs=10):
    print('Starting %s k-means...' % init)
    silhouette_scores = []
    is_forgy = init is 'forgy'
    for num_iterations in range(1, max_iter+1):
        scores = []
        for run in range(num_runs):
            if is_forgy :
                init = data[np.random.choice(data.shape[0], k, replace=False)]
            kmeans = KMeans(n_clusters=k, init=init, max_iter=num_iterations, n_init=1).fit(data_set)
            scores.append(silhouette_score(data_set, kmeans.labels_))
        silhouette_scores.append(scores)
    print('... finished.')
    return silhouette_scores

radius = 5
scaling_factor = 10
generator = DatasetGenerator()
k= 9
num_runs = 30
max_iterations = 20
y_min, y_max = 0.4, 0.7

centers = [Point((i - i % 3), (i % 3)) for i in range(3, 12)]# 3x3 grid
centers = scale_center_points(centers, scaling_factor=scaling_factor)

data_set = []

for center in centers:
    data_set.append(generator.circle_dataset(center, radius, num=200))

data_set = np.concatenate(data_set)

print('Check if data was generated correctly!')
plot_data_set(data_set)

scores_kplusplus = get_silhouette_scores(data_set, init='k-means++')
scores_random = get_silhouette_scores(data_set, init='random')
scores_forgy = get_silhouette_scores(data_set, init='forgy')

avg_plus = [np.average(scores_kplusplus[i]) for i in range(max_iterations)]
std_plus = [np.std(scores_kplusplus[i]) for i in range(max_iterations)]

avg_random = [np.average(scores_random[i]) for i in range(max_iterations)]
std_random = [np.std(scores_random[i]) for i in range(max_iterations)]

avg_forgy = [np.average(scores_forgy[i]) for i in range(max_iterations)]
std_forgy = [np.std(scores_forgy[i]) for i in range(max_iterations)]

iter = [(i+1) for i in range(max_iterations)]

fig = plt.figure()
fig.suptitle("Silhouette scores")

sub = fig.add_subplot(221)
sub.set_title('K-Means++')
sub.errorbar(iter, avg_plus, yerr=std_plus, fmt='bo')
sub.set_ylim([y_min, y_max])

sub = fig.add_subplot(222)
sub.set_title('Random sampled')
sub.errorbar(iter, avg_random, yerr=std_random, fmt='bo')
sub.set_ylim([y_min, y_max])

sub = fig.add_subplot(223)
sub.set_title('Forgy')
sub.errorbar(iter, avg_forgy, yerr=std_forgy, fmt='bo')
sub.set_ylim([y_min, y_max])

sub = fig.add_subplot(224)
sub.set_title('Random partitions')
sub.set_ylim([y_min, y_max])

plt.show()
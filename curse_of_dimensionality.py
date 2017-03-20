import numpy as np
import matplotlib.pyplot as plt

CUBE_SIZE = 2.0
BALL_RADIUS = 1.0

def generate_d_dimensional_point(d, boundry=CUBE_SIZE):
    return [np.random.uniform((-1) * boundry, boundry) for i in xrange(d)]

def generate_n_d_dimensional_points(n, d):
    return [generate_d_dimensional_point(d) for i in xrange(n)]

def how_many_points_in_ball_in_percent(points):
    num_points_in_cube = float(len(points))
    points_in_ball = [p for p in points if np.linalg.norm(p) < BALL_RADIUS]
    num_points_in_ball = len(points_in_ball)
    return 100 * num_points_in_ball / num_points_in_cube

def std_and_average_distance_between_points(points):
    distances = [np.linalg.norm(np.array(p_1)-np.array(p_2)) for p_1 in points for p_2 in points]
    distances = list(filter(lambda a: a != 0.0, distances))
    return (np.average(distances), np.std(distances))

def plot_average_distance_and_std_of_points_in_cube(num_dimensions=10,
                                           num_points=1000, num_runs=10):
    dims = range(1, num_dimensions)
    ratio_average = []
    ratio_std = []

    for dim in xrange(1, num_dimensions):
        points = generate_n_d_dimensional_points(num_points, dim)
        runs = [std_and_average_distance_between_points(points) for run in xrange(num_runs)]
        runs_std = np.array([float(r[0]) for r in runs])
        runs_average = np.array([float(r[1]) for r in runs])

        std_to_average_ratio = runs_std / runs_average

        ratio_std.append(np.std(std_to_average_ratio))
        ratio_average.append(np.average(std_to_average_ratio))
    plt.figure()
    plt.errorbar(dims, ratio_average, yerr=ratio_std, fmt='o')
    plt.ylabel("Std to average ratio")
    plt.xlabel("Dimension")
    plt.show()

def plot_how_many_points_in_ball_on_average(num_dimensions=10,
                                           num_points=10000, num_runs=10):
    dims = range(1, num_dimensions)
    averaged_ball_share = []
    ball_share_std = []

    for dim in xrange(1, num_dimensions):
        points = generate_n_d_dimensional_points(num_points, dim)
        runs = [how_many_points_in_ball_in_percent(points) for run in xrange(num_runs)]
        averaged_ball_share.append(np.average(runs))
        ball_share_std.append(np.std(runs))

    plt.figure()
    plt.errorbar(dims, averaged_ball_share, yerr=ball_share_std, fmt='o')
    plt.ylabel("Percent of points inside the n-hyperball")
    plt.xlabel("Dimension")
    plt.show()



plot_how_many_points_in_ball_on_average()
plot_average_distance_and_std_of_points_in_cube()

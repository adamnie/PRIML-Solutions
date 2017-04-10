import numpy as np


class DatasetGenerator():

    def ring_dataset(self, center,  small_radius, big_radius, num=1000, std=0.1):
        theta = np.linspace(0, 2 * np.pi, num)
        r = np.random.rand(num) * (big_radius - small_radius) + small_radius
        x, y = r * np.cos(theta), r * np.sin(theta)
        return np.column_stack((x + center.x, y + center.y))

    def circle_dataset(self, center, radius, num=1000, std=0.1):
        theta = np.linspace(0, 2 * np.pi, num)
        r = np.random.rand(num) * radius
        x, y = r * np.cos(theta), r * np.sin(theta)
        return np.column_stack((x + center.x, y + center.y))

    def linear_dataset(self, center, slope, num=1000, std=0.1):
       x = np.arange(int(center.x - num/2), int(center.x + num/2))
       delta = np.random.uniform(-std*num, std*num, size=(num,))
       y = slope * x + delta
       return np.column_stack((x + center.x, y + center.y))

class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y
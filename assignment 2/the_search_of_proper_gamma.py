from sklearn.decomposition import KernelPCA
from matplotlib import pyplot as plt

#const

x_ax = 0
y_ax = 1

class GammaSearcher():

    def __init__(self, gamma_start, gamma_step, dataset, num=1000):
        self.gamma_step = gamma_step
        self.dataset = dataset
        self.gamma = gamma_start
        self.kernels = ['rbf', 'cosine', 'sigmoid']
        self.num = num

    def next(self):

        kpca_rbf = KernelPCA(kernel='rbf', n_components=2, gamma=self.gamma)
        kpca_cosine = KernelPCA(kernel='cosine', n_components=2, gamma=self.gamma)
        kpca_sigmoid = KernelPCA(kernel='sigmoid', n_components=2, coef0=self.gamma)

        rbf = kpca_rbf.fit_transform(self.dataset)
        cosine = kpca_cosine.fit_transform(self.dataset)
        sigmoid = kpca_sigmoid.fit_transform(self.dataset)

        figure = plt.figure()
        figure.suptitle("Gamma= %.2f" % self.gamma)
        subplot = figure.add_subplot(311)
        subplot.set_title("rbf")
        subplot.plot(rbf[self.num:, x_ax], rbf[self.num:, y_ax], 'or')
        subplot.plot(rbf[:self.num, x_ax], rbf[:self.num, y_ax], '+b')

        subplot = figure.add_subplot(312)
        subplot.set_title("cosine")
        subplot.plot(cosine[self.num:, x_ax], cosine[self.num:, y_ax], 'or')
        subplot.plot(cosine[:self.num, x_ax], cosine[:self.num, y_ax], '+b')

        subplot = figure.add_subplot(313)
        subplot.set_title("sigmoid")
        subplot.plot(sigmoid[self.num:, x_ax], sigmoid[self.num:, y_ax], 'or')
        subplot.plot(sigmoid[:self.num, x_ax], sigmoid[:self.num, y_ax], '+b')

        self.gamma = self.gamma + self.gamma_step

    def run(self, num_times=10):

        for i in range(num_times):
            self.next()

        plt.show()

""" Module responsible for aggregate the data used for tests/training.
If some dataset is used, it's specific class (with all the necessary preprocessing)
shall be done here and ready for use.
"""

import numpy as np
import matplotlib.pyplot as plt

class Spiral:
    """ This class is inspired by sentdex (https://gist.github.com/Sentdex/454cb20ec5acf0e76ee8ab8448e6266c) and (https://cs231n.github.io/neural-networks-case-study/)"""
    def __init__(self, points, classes):
        self.shuffle_data = True

        X = np.zeros((points*classes, 2))    
        y = np.zeros(points*classes, dtype='uint8')
        for class_number in range(classes):
            ix = range(points*class_number, points*(class_number+1))
            r = np.linspace(0.0, 1, points)  # radius
            t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
            X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
            y[ix] = class_number
        self.X = X
        self.Y = y

        if self.shuffle_data:
            self.shuffle()


    def test(self):
        """Plots the generated points with different color for each class"""
        plt.scatter(self.X[:,0], self.X[:,1], c=self.Y)
        plt.show()
    
    def shuffle(self):
        x_shuf = []
        y_shuf = []
        idx = np.arange(len(self.X))
        np.random.shuffle(idx)
        for i in idx:
            x_shuf.append(self.X[i])
            y_shuf.append(self.Y[i])
        self.X = np.array(x_shuf)
        self.Y = np.array(y_shuf)

    @property
    def data(self):
        return (self.X, self.Y)

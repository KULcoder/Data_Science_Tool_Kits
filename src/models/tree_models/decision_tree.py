import numpy as np

# TODO: 1. add more criterion
#       2. add pruning methods after training
#       3. enable support for categorical features 
#           (problem: how to identify if one feature is categorical?)

class node:
    # the node class to represent decisions and leaf nodes
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, is_leaf=False):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.is_leaf = is_leaf

class criterion:
    # the criterion class to decide best split
    def __init__(self, type):
        self.type = type

    def __call__(self, y):
        if self.type == 'gini':
            return self.gini(y)
        elif self.type == 'variance':
            return self.variance(y)
        
    def gini(self, y):
        # calculate gini impurity for multi-class classification
        # y: numpy array of shape (n_samples, )
        # return: gini impurity

        # TODO: this might be a slow implementation, use numpy all the way might be faster

        n = len(y)
        if n == 0:
            return 0
        classes = np.unique(y)
        gini = 0
        for c in classes:
            gini += (np.sum(y == c) / n) ** 2
        return 1 - gini
    
    def variance(self, y):
        # calculate variance for regression
        # y: numpy array of shape (n_samples, )
        # return: variance
        return np.var(y)
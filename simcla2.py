from collections import Counter
import numpy as np
import pandas as pd
from scipy.spatial import distance
class MahalanobisKNN:
    def __init__(self, k=3):
        self.k = k
        self.x_train = None
        self.y_train = None
        self.cov_i = None
    
    def fit(self, x_train, y_train):
        if isinstance(x_train, pd.DataFrame):
            x_train = x_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
            
        self.x_train = x_train
        self.y_train = y_train
        cov = np.cov(x_train.T)
        self.cov_i = np.linalg.inv(cov)
    
    def predict(self, x_test):
        if isinstance(x_test, pd.DataFrame):
            x_test = x_test.values
            
        predictions = []
        for i in range(len(x_test)):
            distances = []
            for j in range(len(self.x_train)):
                d = distance.mahalanobis(x_test[i], self.x_train[j], self.cov_i)
                distances.append((self.y_train[j], d))
            distances = sorted(distances, key=lambda x: x[1])[:self.k]
            class_counts = Counter([label for label, _ in distances])
            predictions.append(class_counts.most_common(1)[0][0])
        return np.array(predictions)

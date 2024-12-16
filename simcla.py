import numpy as np
from scipy.spatial import distance
class MahalanobisClassifier:
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.cov_i = None
    
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        cov = np.cov(x_train.T)
        self.cov_i = np.linalg.inv(cov)
    
    def predict(self, x_test):
        predictions = []
        for i in range(len(x_test)):
            tmp_mrv = []
            for j in range(len(self.x_train)):
                tmp_mrv.append([self.y_train[j], distance.mahalanobis(x_test[i], self.x_train[j], self.cov_i)])
            tmp_mrv = sorted(tmp_mrv, key=lambda x: x[1])
            predictions.append(tmp_mrv[0][0])  # 最も近い点のクラスを返す
        return np.array(predictions)
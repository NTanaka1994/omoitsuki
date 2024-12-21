from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class DNNClassifier():
    def __init__(self, hidden_layer_sizes=(100,)):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.x_train = None
        self.y_train = None
        self.model = None
        self.pred = None
        self.history = None
    
    def fit(self, x_train, y_train):
        if isinstance(x_train, pd.DataFrame):
            x_train = x_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        self.x_train = x_train
        self.y_train = y_train
        self.model = Sequential()
        for i in range(len(self.hidden_layer_sizes)):
            if i == 0:
                self.model.add(Dense(self.hidden_layer_sizes[i], input_dim=x_train.shape[1], activation="relu"))
            else:
                self.model.add(Dense(self.hidden_layer_sizes[i], activation="relu"))
        self.model.add(Dense(len(list(set(y_train))), activation="softmax"))
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        x_train, x_val, y_train, y_val = train_test_split(self.x_train, self.y_train, test_size=0.3)
        history = self.model.fit(x_train, y_train, epochs=200, batch_size=32, validation_data=(x_val, y_val))
    
    def predict(self, x_test):
        if isinstance(x_test, pd.DataFrame):
            x_test = x_test.values
        self.pred = self.model.predict(x_test)
        y_pred = np.argmax(self.pred, axis=1)
        return y_pred

    def predict_proba_(self, x_test):
        if isinstance(x_test, pd.DataFrame):
            x_test = x_test.values
        return self.model.predict(x_test)

class DNNRegressor():
    def __init__(self, hidden_layer_sizes=(100,)):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.x_train = None
        self.y_train = None
        self.model = None
        self.pred = None
        self.history = None
    
    def fit(self, x_train, y_train):
        if isinstance(x_train, pd.DataFrame):
            x_train = x_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        self.x_train = x_train
        self.y_train = y_train
        self.model = Sequential()
        for i in range(len(self.hidden_layer_sizes)):
            if i == 0:
                self.model.add(Dense(self.hidden_layer_sizes[i], input_dim=x_train.shape[1], activation="relu"))
            else:
                self.model.add(Dense(self.hidden_layer_sizes[i], activation="relu"))
        self.model.add(Dense(1, activation="linear"))
        self.model.compile(optimizer="adam", loss="mean_squared_error")
        x_train, x_val, y_train, y_val = train_test_split(self.x_train, self.y_train, test_size=0.3)
        history = self.model.fit(x_train, y_train, epochs=200, batch_size=32, validation_data=(x_val, y_val))
    
    def predict(self, x_test):
        if isinstance(x_test, pd.DataFrame):
            x_test = x_test.values
        self.pred = self.model.predict(x_test)
        return self.pred

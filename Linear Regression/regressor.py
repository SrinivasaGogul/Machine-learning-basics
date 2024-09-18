from tkinter.ttk import Treeview

import  pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
# from scipy.special import y_pred
from sklearn.model_selection import train_test_split



class LinearRegression:

    def __init__(self, lr, epochs):

        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None
        np.random.seed(42)

    def fit(self, X_train, y_train):

        n_samples, n_features = X_train.shape
        print(f"the number of features are ----> {n_features}")
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            y_pred = np.dot(X_train,self.weights) + self.bias

            dw = (2/n_samples) * np.dot(X_train.T, (y_train - y_pred))
            db = (2/n_samples) * np.sum((y_train - y_pred))

            self.weights = self.weights - (self.lr * dw)
            self.bias = self.bias - (self.lr * db)

    def predict(self, x):
        y_pred = np.dot(x, self.weights) + self.bias
        return y_pred


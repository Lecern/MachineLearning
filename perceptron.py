import numpy as np


class Perceptron():
    def __init__(self, eta=0.01, n_iter=10):
        """
        eta 学習率 float
        n_iter トレーニングデータのトレーニング回数
        """
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """

        :param X: 配列の様なデータ構造  shape=[n_samples, n_features]
        :param y: 配列の様なデータ構造  shape=[n_samples]
        :return:
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                # 重み w1...wm の更新
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                # 重み w0
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:] + self.w_[0])

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

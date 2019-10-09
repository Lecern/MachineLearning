import numpy as np


class AdalineGD(object):

    # eta: float 学習率（0.0より大きく1.0以下の値）
    # n_iter: int トレーニングデータのトレーニング回数
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        トレーニングデータに適合させる
        :param X: 　{配列のようなデータ構造} shape=[n_samples, n_features]
                    トレーニングデータ
        :param y: 　配列のようなデータ構造 shape=[n_samples]
                    目的変数
        :return: self: object
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            # 活性化関数の出力の計算
            output = self.net_input(X)
            # 誤差の計算yi-Φ(zi)の計算
            errors = y - output
            # w1,...,wmの更新
            self.w_[1:] += self.eta * X.T.dot(errors)
            # w0の更新
            self.w_[0] += self.eta * errors.sum()
            # コスト関数の計算
            cost = (errors ** 2).sum() / 2.0
            # コストの格納
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """総入力の計算"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """線形活性化関数の出力を計算"""
        return self.net_input(X)

    def predict(self, X):
        """１ステップ後のクラスラベルを返す"""
        return np.where(self.activation(X) >= 0.0, 1, -1)
import numpy as np
from numpy.random import seed


class AdalineSGD(object):
    def __init__(self, n_iter=10, eta=0.01, shuffle=True, random_state=None):
        """
        :param n_iter: トレーニングデータのトレーニング回数
        :param eta: 学習率
        :param shuffle: 循環を回避するために各エポックでトレーニングデータをシャッフル
        :param random_state: シャッフルに使用するランダムステートを設定し、重みを初期化
        """
        self.n_iter = n_iter
        self.eta = eta
        # 重みの初期化フラグはfalseに設定
        self.w_initialized = False
        # 各エポックでトレーニングデータをシャフルするかどうかのフラグを初期化
        self.shuffle = shuffle
        # 引数random_stateが指定された場合は乱数種を設定
        if random_state:
            seed(random_state)

    def fit(self, X, y):
        """
        トレーニングデータに適合させる
        :param X: トレーニングデータ shape = [n_samples, n_features]
        :param y: 目的変数 shep = [n_samples]
        :return: self: object
        """
        # 重みベクトルの生成　X.shape[1] -> Xの列数 -> 特徴量の個数
        self._initialize_weight(X.shape[1])
        # コストを格納するリストの生成
        self.cost_ = []
        # トレーニング回数分トレーニングデータを反復
        for i in range(self.n_iter):
            # 指定さてた場合はトレーニングデータをシャッフル
            if self.shuffle:
                X, y = self._shuffle(X, y)
            # 各サンプルのコストを格納するリストの生成
            cost = []
            # 各サンプルに対する計算
            for xi, target in zip(X, y):
                # 特徴量xiと目的変数yを用いた重みの更新とコストの計算
                cost.append(self._update_weight(xi, target))
            # サンプルの平均コストの計算
            avg_cost = sum(cost) / len(y)
            # 平均コストを格納
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """重みを最初期化することなくトレーニングに適合させる"""
        # 初期化されていない場合は初期化を実行
        if not self.w_initialized:
            self._initialize_weight(X.shape[1])
        # 目的変数yの要素数が2以上の場合は各サンプルの特徴量xiと目的変数yで重みを更新
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weight(xi, target)
        # 目的変数yの要素数が1の場合はサンプル全体の特徴量Xと目的変数yで重みを更新
        else:
            self._update_weight(X, y)
        return self

    def _shuffle(self, X, y):
        """"トレーニングデータをシャッフル"""
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initialize_weight(self, m):
        """重みを0に初期化"""
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True

    def _update_weight(self, xi, target):
        """ADALINEの学習規則を用いて重みを更新"""
        # 活性化関数の出力の計算
        output = self.net_input(xi)
        # 誤差の計算
        error = target - output
        # 重みw1,...,wmの更新
        self.w_[1:] += self.eta * xi.dot(error)
        # 重みw0の更新
        self.w_[0] += self.eta * error
        # コストの計算
        cost = 0.5 * error ** 2
        return cost

    def net_input(self, X):
        """総入力を計算"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """線形活性化関数の出力を計算"""
        return self.net_input(X)

    def predict(self, X):
        """1ステップ後のクラスラベルを返す"""
        return np.where(self.activation(X) >= 0.0, 1, -1)

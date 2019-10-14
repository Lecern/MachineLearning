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
        pass

    def partial_fit(self, X, y):
        pass

    def _shuffle(self, X, y):
        pass

    def _initialize_weight(self, m):
        pass

    def _update_weight(self, xi, target):
        pass

    def net_input(self, X):
        pass

    def activation(self, X):
        pass

    def predict(self, X):
        pass

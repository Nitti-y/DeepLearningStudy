import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
from dataset.mnist import load_mnist

def main():
    x, t = getData()
    network = initNetwork()

    accuracyCnt = 0
    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y) # 最も確率の高い要素のインデックスを取得
        if p == t[i]:
            accuracyCnt += 1

    print("Accuracy:" + str(float(accuracyCnt) / len(x)))

def getData():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize = True, flatten = True, one_hot_label = False)
    
    return x_test, t_test

def initNetwork():
    """サンプルの学習済みパラメータを読み込む
    
    ニューラルネットの構成
    ----------
    入力層：784個
    出力層：10個
    隠れ層：2つ（1つ目：50個，2つ目：100個）

    各パラメータの仕様
    ----------
    W1：サイズ784*50の行列
    b1：サイズ50のベクトル
    W2：サイズ50*100の行列
    b2：サイズ100のベクトル
    W3：サイズ100*10の行列
    b3：サイズ10のベクトル
    """
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y
from math import e
from math import ceil
from math import floor

def train_test_split (base, perc_train = 0.7, perc_test = 0.3):
    target = ['class']

    size_train = ceil(base.shape[0] * perc_train)
    train = base.sample(n = size_train)
    
    x_train = train.drop(columns = target).to_numpy()
    y_train = train.filter(target).to_numpy()

    size_test = floor(base.shape[0] * perc_test)
    test = base.sample(n = size_test)

    x_test = test.drop(columns = target).to_numpy()
    y_test = test.filter(target).to_numpy()

    #retorna os conjuntos de treino e teste como np_arrays
    return x_train, y_train, x_test, y_test

def f(net): #sigmoide
    return 1 / (1 + e ** net)

def df(f_net): #recebe o valor de f direto
    return f_net * (1 - f_net)

def tanh(net):
    return (1 - e ** (-2 * net)) / (1 + e ** (-2 * net))

def d_tanh(tanh):
    return 1 - tanh ** 2
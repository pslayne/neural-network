from math import e
from math import ceil
from math import floor
import numpy as np
import auxiliar

class MLP:
    def __init__(self, base, i, j, k):
        self.base = base
        self.i = i #nº de neurônios de entrada
        self.j = j #nº de neurônios da camada oculta
        self.k = k #nº de neurônios de saída
        
        #inicialização dos pesos da camada oculta (sinapses com a camada de entrada)
        self.hidden = np.random.uniform(low=-1.0, high=1.1, size=(j, i)) 

        #inicialização dos pesos da camada de saída (sinapses com a camada oculta)
        self.out = np.random.uniform(low=-1.0, high=1.1, size=(k, j)) 

    #cálculo dos net's da camada oculta
    def hidden_iteration(self, x): #recebe a matriz de sinapses e um valor de entrada
        net_h = 0
        for a in range (0, self.j): #p cada neurônios da camada oculta
            for b in range(self.i):
                net_h += self.hidden[a][b] * x[b] #somatório
        
        #aplica a função de transferência 
        return auxiliar.tanh(net_h)

    #cálculo dos net's da camada de saída
    def out_iteration(self, x): #recebe a matriz de sinapses e um valor de entrada (que vem da camada oculta)
        net_o = 0
        
        for a in range (0, self.k): #p cada neurônios da camada oculta
            for b in range(self.j):
                net_o += self.out[a][b] * x #somatório
        
        #aplica a função de transferência 
        return auxiliar.tanh(net_o)

    def out_errors(self, o_y, d_y):
        delta = []
        
        for a in range (0, len(d_y)): #p cada neurônios da camada de saída
            delta.append((d_y[a] - o_y) * auxiliar.d_tanh(o_y)) #(valor desejado - valor obtido) * derivada da função de transferência aplicada à saída obtida

        return delta  

    def hidden_errors(self, h_x, delta_out):
        delta = []
        ac = 0
        
        for a in range (0, self.j): #p cada neurônios da camada oculta
            for b in range (0, self.k):
                ac = ac + self.out[b][a] * delta_out[0]
            
            d = auxiliar.d_tanh(h_x) * ac
            delta.append(d)
            ac = 0

        return delta

    def net_error(self, delta_out):
        ac = 0
        for a in range (0, len(delta_out)):
            ac = ac + delta_out[a] ** 2
        return (ac) / 2 

    def out_update(self, delta_out, h_x, n): #deltas de saída, resultados da camada oculta, taxa de aprendizagem
        for a in range (0, self.k):
            for b in range (0, self.j):
                self.out[a][b] = self.out[a][b] + (n * delta_out[0] * h_x)

    def hidden_update(self, delta_hidden, x, n): #deltas da camada oculta, entradas, taxa de aprendizagem
        for a in range (0, self.j):
            for b in range (0, self.i):
                self.hidden[a][b] = self.hidden[a][b] + (n * delta_hidden[a] * x[b])

    def fit(self, x_train, y_train, iterations = 1000, n = 0.4):
        smaller_error = 1
        it = 0
        while it < iterations:
            #smaller_error = 1
            for a in range(0, x_train.shape[0]):
                #forward feeding
                hidden_output = self.hidden_iteration(x_train[a]) 
                output = self.out_iteration(hidden_output)

                #backpropagation
                delta_out = self.out_errors(output, y_train[a])
                delta_hidden = self.hidden_errors(hidden_output, delta_out)

                #erro da rede
                error = self.net_error(delta_out)
                smaller_error = error if error < smaller_error else smaller_error

                #atualiza os pesos
                self.out_update(delta_out, hidden_output, n)
                self.hidden_update(delta_hidden, x_train[a], n)
            
            # if it % 100 == 0:
            #     print(smaller_error)

            it = it + 1

    def predict(self, x_test):
        result = []
        for a in range(0, x_test.shape[0]):
            #calcula os net e já passa pela função de transferência / retornam listas
            hidden_output = self.hidden_iteration(x_test[a]) 
            output = self.out_iteration(hidden_output)

            result.append(round(output, 10))
        return result

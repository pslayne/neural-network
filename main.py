import pandas as pd
from MLP import MLP
from auxiliar import train_test_split

i = 25 #nº de neurônios de entrada
j = 10 #nº de neurônios da camada oculta
k = 3 #nº de neurônios de saída

#base de dados
sheet_url = "https://docs.google.com/spreadsheets/d/10Bd1gwY9GK6dJg-web9TLN0PSJ_m4p820BngimA-ud0/edit#gid=0"
url = sheet_url.replace('/edit#gid=', '/export?format=csv&gid=')
base = pd.read_csv(url) #os números estão representados por conjuntos de 25 bits

mlp = MLP(base, i, j, k)

x_train, y_train, x_test, y_test = train_test_split(base)
mlp.fit(x_train, y_train)
pred = mlp.predict(x_test)
print('y_test')
print(y_test)
print('predict')
print(pred)




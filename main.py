import numpy as np
#Тема 11 Создайте нейронную сеть, которая правильно классифицирует объекты, пользуясь данными из табл. 8.
#Ответ: Если «Параметр 1» = 1, то 1-й класс, если «Параметр 2» =
#«Параметр 3», то 2-й класс, в противном случае – 3-й класс.
#Вспомогательные функции
def Sigmoid(Object):                        #сигмоида
  return 1 / (1 + np.exp(-Object))

def Sigmoid_derivative(Object):                    #производная сигмоиды
  return Sigmoid(Object) * (1 - Sigmoid(Object))

def Y_Vector(y):                 #перевод ответа в бинарный вид
  y_full = np.zeros((len(y), 3))
  for id, y_id in enumerate(y):
    y_full[id, y_id - 1] = 1
  return y_full
#Входные значения
x = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1], [0, 1, 0], [0, 1, 1], [0, 0, 1], [0, 1, 0], [1, 1, 1], [0, 0, 0]])
y = np.array([1, 1, 2, 3, 2, 3, 3, 1, 2])
# Алгоритм
Speed = 0.25  # скорость обучения
number_epochs = 1000
weight1 = np.random.rand(3, 3)  # веса первого слоя
b1 = np.random.rand(1, 3)  # смещения первого слоя

weight2 = np.random.rand(3, 3)  # веса второго слоя
b2 = np.random.rand(1, 3)  # значения второго слоя

for epoch in range(number_epochs):
    Object = x @ weight1 + b1  # промежуточные значения (t - temp)
    h1 = Sigmoid(Object)  # значения функции активации от t1

    tout = h1 @ weight2 + b2
    hout = Sigmoid(tout)

    y_full = Y_Vector(y)  # выходные значения

    # Вычисление градиента
    de_dtout = hout - y_full  # ошибка последнего слоя
    de_dwout = h1.T @ de_dtout
    de_dbout = np.sum(de_dtout, axis=0, keepdims=True)  # смещения для выходного слоя

    de_dh1 = de_dtout @ weight2.T
    de_dObject = de_dh1 * Sigmoid_derivative(Object)
    de_dw1 = x.T @ de_dObject
    de_db1 = np.sum(de_dObject, axis=0, keepdims=True)  # смещения для скрытого слоя

    # Градиентный спуск
    weight1= weight1 - Speed * de_dw1
    b1 = b1 - Speed * de_db1
    weight2= weight2 - Speed * de_dwout
    b2 = b2 - Speed * de_dbout

    counter = 0
    for i in range(len(x)):
        Object = x[i] @ weight1 + b1
        h1 = Sigmoid(Object)
        tout = h1 @ weight2 + b2
        hout = Sigmoid(tout)

        if np.argmax(hout) + 1 == y[i]: counter += 1  # подсчет верных выводов
    if (epoch + 1) % 100 == 0: print('Точность на эпохе', (epoch + 1), ' = ', counter / 9)
    if counter / 9 == 1.0:
        print('Точность на эпохе', (epoch + 1), ' = ', counter / 9)
        break
Object = [0, 0, 1] @ weight1 + b1
h1 = Sigmoid(Object)

tout = h1 @ weight2 + b2
hout = Sigmoid(tout)
print(np.argmax(hout) + 1)


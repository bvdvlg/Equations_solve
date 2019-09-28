import pandas as pd
import numpy as np


def find_nonzero(i, m, e):
    if m[i].loc[i] != 0:
        return 0
    if np.delete(m[i].loc[i:len(m.index)].unique(), 0).size == 0:
        return 1
    e[i].loc[[k for k in m[i].loc[i:len(m.index)].index if m[i].loc[k] != 0][0]] += 1
    m.iloc[i] += m.loc[[k for k in m[i].loc[i:len(m.index)].index if m[i].loc[k] != 0][0]]
    return 0


def read_inp(filename):
    f = open(filename, "r")
    pars = [line for line in f]
    k = int(pars.pop(0))
    for i in range(len(pars)):
        pars[i] = [float(x) for x in (pars[i].strip('\n')).split(' ') if x!= '']
    return pd.DataFrame(pars), k


def change(i, j, m, e):
    e[i].loc[j] += (m[i].loc[j]/m[i].loc[i])
    m.loc[j] += (m.loc[i])*(m[i].loc[j]/m[i].loc[i])*(-1)
    return


name = "second.txt"
matrix, n = read_inp(name)
A = matrix.copy()
print("Начальная матрица:")
print(matrix)
E = pd.DataFrame([[0.0 for t in range(n)] for x in range(n)])
for i in range(n): E[i].loc[i] = 1
for i in range(n):
    term = find_nonzero(i, matrix, E)
    if term != 1:
        for j in range(i+1, len(matrix.index)):
            change(i, j, matrix, E)
print("Матрица L:")
print(E)
L = E
print("Матрица U:")
print(matrix)
U = matrix
b = np.matrix(A).dot(np.array([x for x in range(1, n+1)]))
print(b)
A[4] = (b.T)
b = A[4]
y = [0 for i in L.columns]
for elem in L.index:
    y[elem] = b[elem]/L[elem].loc[elem]
    b = b - y[elem]*L[elem]
    L[elem] = pd.Series([0 for i in L.index])
print ("Получившееся значение y: ", end='')
print(y)
y1 = y
x = [0 for i in U.columns]
for elem in U.index[::-1]:
    x[elem] = y[elem]/U[elem].loc[elem]
    y = y - x[elem]*U[elem]
    U[elem] = pd.Series([0 for i in U.index])
y = y1
print ("Искомое значение x: ", end='')
print(x)
x = np.array(x)
y = np.array(y)
print(x-y)
print("Норма разницы между векторами x и y равна: ")
print("Евклидова: ")
print((x-y).dot((x-y).T)**0.5)
print("Кубическая: ")
print(pd.Series(map(abs, x-y)).sum())
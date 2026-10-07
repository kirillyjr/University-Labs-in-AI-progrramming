import numpy as np
import time


data = np.random.rand(24 * 60, 12)   # случайные данные: строка - минута, столбец - датчик

# 1.1 переформатирование: 1440 минут -> 24 часа по 60 минут
d = data.reshape(24, 60, 12)         # нарезаем на (час, минута, датчик), числа те же
print(d.shape, np.array_equal(d.reshape(-1, 12), data))   # True: порядок не нарушен

# 1.2 средние
print(d.mean(axis=(1, 2)).shape)   # по часам: (24,)
print(d.mean(axis=(0, 1)).shape)   # по датчикам: (12,)
print(d.mean(axis=1).shape)        # час и датчик: (24, 12), усредняем только минуты

# 1.3 f(x) = x / (1 + |x|)
f = d / (1 + np.abs(d))            # формула сразу ко всем числам, без цикла
print(f.shape == d.shape)          # форма не меняется (поэлементная операция)
print(np.abs(f - d).max())         # максимальное изменение


np.random.seed(0)
X = np.random.rand(500, 5)         # 500 точек, у каждой 5 координат

# 2.1 центрирование: (500,5) - (5,) -> (500,5)
center = X.mean(axis=0)            # центр масс: среднее по каждой координате, (5,)
Xc = X - center                    # вычитаем центр из каждой точки (broadcasting)
print(Xc.mean(axis=0))             # ~0

# 2.2 расстояния, способ 1: broadcasting (500,1,5) - (1,500,5) -> (500,500,5)
t = time.time()
diff = X[:, None, :] - X[None, :, :]       # Вставляем «пустые» оси. None в квадратных скобках создаёт новую ось размера 1
D1 = np.sqrt((diff ** 2).sum(axis=2))      # сумма по признакам -> (500,500)
t1 = time.time() - t 

# способ 2: |x-y|^2 = |x|^2 + |y|^2 - 2xy
t = time.time()
sq = (X ** 2).sum(axis=1)                  # (500,)
D2 = sq[:, None] + sq[None, :] - 2 * X @ X.T   # (500,500): X @ X.T - скалярные произведения всех пар
D2 = np.sqrt(np.maximum(D2, 0))            # maximum защищает от -1e-16 под корнем
t2 = time.time() - t

print(np.allclose(D1, D2, atol=1e-6))   # на диагонали у способа 2 остаётся ~1e-8
print("способ 1:", t1, "c, самый большой массив", diff.nbytes / 1e6, "МБ") #куб diff - 10 мб
print("способ 2:", t2, "c, самый большой массив", D2.nbytes / 1e6, "МБ")

# 2.3 соседи
D = D1.copy()                      # копия, чтобы не портить D1
np.fill_diagonal(D, np.inf)        # иначе ближайшая точка - она сама
nearest = D.argmin(axis=1)         # (500,)  nearest[i] это номер точки, ближайшей к i
far = D1.mean(axis=1).argmax()     # точка с наибольшим средним расстоянием до остальных
print(nearest[:5], far) 


# 3.1 нормализация строк
#Нормализовать значит сделать длину каждой строки равной 1. Для этого каждую строку делят на её длину
X = np.random.rand(1000, 10)
norms = np.linalg.norm(X, axis=1)  # длина каждой строки, (1000,)
Xn = X / norms[:, None]            # делим строку на её длину; [:, None] делает столбец (1000,1)
print(norms.shape, np.allclose(np.linalg.norm(Xn, axis=1), 1))

# 3.2 логическая маска
X = np.random.rand(500, 20)
row_mean = X.mean(axis=1)          # среднее каждой строки, (500,)
col_mean = X.mean(axis=0)          # среднее каждого столбца, (20,)
mask = (X > row_mean[:, None]) & (X < col_mean)   # True там, где больше среднего строки И меньше среднего столбца, (500,20)
X[mask] = 0                        # на месте True пишем 0
print(mask.sum(), X.shape)         # сколько заменено, форма та же

import numpy as np
import time
import tracemalloc

#задание 1
data = np.random.rand(24*60, 12)  # Generate random data with 24*60 samples and 12 features
d = data.reshape(24, 60, 12)  # Reshape the data to have shape (24, 60, 12)
print(d.shape, np.array_equal(d.reshape(-1, 12), data))  


 
print(d.mean(axis=(1, 2)).shape)                 # по часам: (24,)
print(d.mean(axis=(0, 1)).shape)                 # по датчикам: (12,)
print(d.mean(axis=1).shape) 

f = d / (1 + np.abs(d))
print(f.shape == d.shape)                        # форма не меняется
print(np.abs(f - d).max())                       # макс. изменение


#задание 2
import numpy as np
import time

np.random.seed(0)
X = np.random.rand(500, 5)

# 2.1 Центрирование
center = X.mean(axis=0)        # центр масс, (5,)
Xc = X - center                # (500,5) - (5,): вычитается из каждой строки
print(Xc.mean(axis=0))         # ~0 во всех 5 координатах

# 2.2 Расстояния, способ 1: broadcasting
# (500,1,5) - (1,500,5) -> (500,500,5), сумма по оси 2 (признаки) -> (500,500)
tracemalloc.start()
t = time.time()
diff = X[:, None, :] - X[None, :, :]
D1 = np.sqrt((diff ** 2).sum(axis=2))
t1 = time.time() - t
mem1 = tracemalloc.get_traced_memory()[1] / 1e6   # пик памяти, МБ
tracemalloc.stop()
del diff

# способ 2: |x-y|² = |x|² + |y|² - 2xy
# sq: (500,), sq[:,None]+sq[None,:]: (500,500), X @ X.T: (500,5)@(5,500)=(500,500)
tracemalloc.start()
t = time.time()
sq = (X ** 2).sum(axis=1)
D2 = sq[:, None] + sq[None, :] - 2 * X @ X.T
D2 = np.sqrt(np.maximum(D2, 0))                   # maximum: защита от -1e-16
t2 = time.time() - t
mem2 = tracemalloc.get_traced_memory()[1] / 1e6
tracemalloc.stop()

print(np.allclose(D1, D2, atol=1e-6))
print(f"способ 1: {t1:.4f} c, пик памяти {mem1:.1f} МБ")
print(f"способ 2: {t2:.4f} c, пик памяти {mem2:.1f} МБ")

# 2.3 Соседи
D = D1.copy()
np.fill_diagonal(D, np.inf)          # чтобы точка не была ближайшей к себе
nearest = D.argmin(axis=1)           # (500,) индекс ближайшей точки
far = D1.mean(axis=1).argmax()       # точка с наибольшим средним расстоянием
print(nearest[:5], far)


#задание 3
X = np.random.rand(1000, 10)

norms = np.linalg.norm(X, axis=1)
Xn = X / norms[:, None]

print(norms.shape)
print(np.linalg.norm(Xn, axis=1))
print(np.allclose(np.linalg.norm(Xn, axis=1), 1))

# 3.2 Логическая маска
X = np.random.rand(500, 20)

row_mean = X.mean(axis=1)    # (500,)  среднее по каждой строке
col_mean = X.mean(axis=0)    # (20,)   среднее по каждому столбцу
print(row_mean.shape, col_mean.shape)

# (500,20) > (500,1) и (500,20) < (20,) -> булева маска (500,20)
mask = (X > row_mean[:, None]) & (X < col_mean)
print(mask.shape, mask.dtype)

shape_before = X.shape
X[mask] = 0
print("заменено элементов:", mask.sum())
print("форма не изменилась:", X.shape == shape_before, "| нулей в X:", (X == 0).sum())

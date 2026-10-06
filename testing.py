import numpy as np
a = np.array([[1, 2, 3],
              [4, 5, 6]], dtype='float64')   # форма (2, 3)

print(a.mean())   # 3.5
print(a.mean(axis=0))   # [2.5, 3.5, 4.5]   форма (3,)
print(a.mean(axis=1))   # [2.0, 5.0]        форма (2,)
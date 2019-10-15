import numpy as np

randArr = np.random.randn(3, 2)

randArr2 = np.random.randn(3, 2)

print("random array 1:\n", randArr)

print("random array 2:\n", randArr2)

a = ("Apple", "Orange", "Banana")

b = ("Tom", "Jack", "Alice")

print(tuple(zip(a, b)))

for i, j in zip(a, b):
    print("i:", i, "j:", j)

t_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
print(t_list[1:])

print("shape of array1:", randArr.shape[1])

randArr3 = np.random.rand(3, 2, 2)

print("random array 3:\n", randArr3)

print("---------------------------------------------------------------------")

simu_X = np.arange(10).reshape((5, 2))
simu_X = np.random.permutation(simu_X)
print(simu_X)
simu_r = np.random.permutation(5)
print(simu_r)

print("X[r]", simu_X[simu_r])

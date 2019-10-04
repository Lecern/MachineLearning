import numpy as np

randArr = np.random.randn(3, 2)

randArr2 = np.random.randn(3, 2)

print("random array 1:", randArr)

print("random array 2:", randArr2)

a = ("Apple", "Orange", "Banana")

b = ("Tom", "Jack", "Alice")

print(tuple(zip(a, b)))

for i, j in zip(a, b):
    print("i:", i, "j:", j)

print("Hello world")

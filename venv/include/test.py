import torch as t
import numpy as np
import math

a = np.arange(5)
print(a)
print(~a)

b = np.arange(4, -1, -1)

print(np.logical_or(a == b, a > b))

a = 8
print(math.exp(np.log(a)))


print("虎儿+")

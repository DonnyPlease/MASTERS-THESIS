import numpy as np

a1 = 455960398099578816
b1 = -0.198190
a2 = 1588844504510639
b2 = -0.007716

x = np.log(a1/a2)/(b2-b1)
print(x)
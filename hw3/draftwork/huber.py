import matplotlib.pyplot as plt
import numpy as np

delta = 50

def f(y):
	if abs(y) <= delta:
		return 0.5 * y**2
	else:
		return delta * (abs(y) - 0.5 * delta)

y=np.linspace(0, 100, 200) # generate numbers from -100 to 100, generate 200 of them

plt.figure()
f2 = np.vectorize(f)
L = f2(y)
plt.plot(y, L)
plt.plot(y, 0.5*y**2)
plt.show()
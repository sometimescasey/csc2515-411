'''
Not submitted for grades.
'''

import random

total = 0
var_total = 0
trials = 1000000
d = 2

E_Z = 1/6
Var_Z = 7/180

for i in range(0, trials):
	for j in range(0, d):
		# print(i)
		x = random.uniform(0,1)
		y = random.uniform(0,1)

		z = (x - y) ** 2
		total += z
		var_total += (z - E_Z) ** 2

print("Mean: {:0.5f} | Theoretical: {:0.5f} | delta: {:0.5f}".format(
	total / trials, 
	E_Z * d,
	total / trials - E_Z * d))
print("Variance: {:0.5f} | Theoretical: {:0.5f} | delta: {:0.5f}".format(
	var_total / trials, 
	Var_Z*d,
	var_total / trials - Var_Z*d))


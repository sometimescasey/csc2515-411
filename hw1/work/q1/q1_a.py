'''
Not submitted for grades.
'''

import scipy.integrate

def z(y, x):
	return (x-y) ** 2

def var_z(y, x):
	return (z(y,x) - E_Z) ** 2

def z_sq(y, x):
	return (z(y,x)) ** 2

# Expected Z
E_Z, E_Z_err = scipy.integrate.dblquad(
	z, # dYdX
	0, 1, # outside integral 
	lambda x: 0, 
	lambda x: 1, # inside integral 
)

print("E_Z: {}".format(E_Z))

# Variance of Z: 2 ways

# 1st definition: expected value of (Z - E[Z}])^2
Var_Z_1, Var_Z_1_err = scipy.integrate.dblquad(
	var_z, # dYdX
	0, 1,
	lambda x: 0, 
	lambda x: 1, # inside integral
	)

print("Var_Z_1: {}".format(Var_Z_1))

# 2nd definition: E[Z^2] - (E[Z])^2
exp_Z_sq, exp_Z_sq_err = scipy.integrate.dblquad(
	z_sq, # dYdX
	0, 1,
	lambda x: 0, 
	lambda x: 1, # inside integral
	)

Var_Z_2 = exp_Z_sq - (E_Z) ** 2

print("Var_Z_2: {}".format(Var_Z_2))
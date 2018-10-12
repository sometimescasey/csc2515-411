import autograd.numpy as np
import autograd
import matplotlib.pyplot as plt
np.set_printoptions(precision=3)

def huberLoss(W, x, b, y_target, threshold):
	# Make predictions
	y_pred = np.matmul(x.T, W) + b
	# print(y_pred.shape)

	residual = y_pred - y_target

	H = np.where(abs(residual) <= threshold, 
		0.5*residual**2, 
		threshold * abs(residual - 0.5*threshold))

	return H

def get_gd_deltas(f, learning_rate, W, x, b, y_target, threshold):
	df_dW = autograd.elementwise_grad(f,0)
	df_db = autograd.elementwise_grad(f,2)
	
	delta_W = -learning_rate * df_dW(W, x, b, y_target, threshold)
	delta_b = -learning_rate * df_db(W, x, b, y_target, threshold)

	return delta_W, delta_b

def main():
	m = 3 # x1, x2, x3
	N = 100 # 100 samples

	# GD params
	N_ITER = 20
	learning_rate = 0.01
	threshold = 0.5 # Huber threshold
	
	# Generate random m x N matrix to use as X
	x = np.random.rand(m, N)

	# Initialize W to zeros
	W = np.zeros((m,1))

	# Initialize b to zeros
	b = np.zeros((N,1))

	# Dummy y_target: all ones
	y_target = np.zeros((N, 1))
	y_target.fill(1)
	
	for i in range(N_ITER):
		# Allowed to iterate over loops of W and b updates, just not over the training data

		delta_W, delta_b = get_gd_deltas(huberLoss, learning_rate, W, x, b, y_target, threshold)

		W += delta_W
		b += delta_b

		totalLoss = np.sum(huberLoss(W, x, b, y_target, threshold), axis=0)
		print("Iteration {}: Total Huber loss = {}".format(i, totalLoss[0]))

if __name__ == "__main__":
	main()


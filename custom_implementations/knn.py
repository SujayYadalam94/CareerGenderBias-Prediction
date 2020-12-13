import sys
import numpy as np
import pandas as pd
import time
import math
import matplotlib.pyplot as plt

#calculate eucledian distance
def eucledian_distance(x1, x2):
	return np.linalg.norm(x1-x2)

#calculate manhattan distance
def manhattan_distance(x1, x2):
	return np.sum(abs(np.subtract(x1, x2)))

#min max normalization
def min_max_normalization(x, minimum, maximum):
	if (minimum == maximum):
		return 1
	else:
		result = (x-minimum)/(maximum-minimum)
	return result

#calculate distance weight
def calculate_weight(distance):
	return np.exp(-1*(distance**2))

#nearest neighbour calculation
def nearest_neighbour(k, X, Y, Xnew):
	distance = list()

	for i in range(len(X)):
		dist = eucledian_distance(X[i], Xnew)
		distance.append((i, dist))

	distance.sort(key=lambda tup: tup[1])

	sum_survive = 0
	sum_not_survive = 0

	for i in range(k):
		if (Y[int(distance[i][0])][0] == 1):
			sum_survive += calculate_weight(distance[i][1])
		else:
			sum_not_survive += calculate_weight(distance[i][1])

	if (sum_survive > sum_not_survive):
		return 1
	else:
		return 0

#knn n-cross validation
def knn_cross_valid(n, N, X, Y, k):
	accuracy = []

	count0 = np.count_nonzero(Y == 0)
	count1 = np.count_nonzero(Y == 1)

	X1 = np.delete(X, [*range(0, count0)], 0)
	X0 = np.delete(X, [*range(count0, N)], 0)

	Y1 = np.delete(Y, [*range(0, count0)], 0)
	Y0 = np.delete(Y, [*range(count0, N)], 0)

	fold_size0 = count0//n
	fold_size1 = count1//n
	total_fold = fold_size0 + fold_size1

	for i in range(n):
		start_fold0 = i*fold_size0
		end_fold0 = i*fold_size0 + fold_size0

		range_test0 = [*range(start_fold0, end_fold0)]
		range_train0 = [*range(0, start_fold0)]
		range_train0.extend([*range(end_fold0, count0)])

		X0train = np.delete(X0, range_test0, 0)
		Y0train = np.delete(Y0, range_test0, 0)
		X0test = np.delete(X0, range_train0, 0)
		Y0test = np.delete(Y0, range_train0, 0)

		start_fold1 = i*fold_size1
		end_fold1 = i*fold_size1 + fold_size1

		range_test1 = [*range(start_fold1, end_fold1)]
		range_train1 = [*range(0, start_fold1)]
		range_train1.extend([*range(end_fold1, count1)])

		X1train = np.delete(X1, range_test1, 0)
		Y1train = np.delete(Y1, range_test1, 0)
		X1test = np.delete(X1, range_train1, 0)
		Y1test = np.delete(Y1, range_train1, 0)

		Xtrain = np.append(X0train, X1train, 0)
		Ytrain = np.append(Y0train, Y1train, 0)
		Xtest = np.append(X0test, X1test, 0)
		Ytest = np.append(Y0test, Y1test, 0)

		correct = 0
		for j in range(0, total_fold):
			prediction = nearest_neighbour(k, Xtrain, Ytrain, Xtest[j])
			if (Ytest[j][0] == prediction):
				correct += 1
		accuracy.append(correct/total_fold)
	return accuracy

def main():

	if len(sys.argv) < 2:
		print("Usage: python3 knn.py input_file")
		exit(1)

	df = pd.read_csv(sys.argv[1])

    # Remove lines without values
	df.dropna(how='any', inplace=True)

	df.sort_values('dscore')

	# separate data into samples and output classes
	x = df.drop(columns=['dscore'])
	y = df.loc[:,['dscore']]#.to_numpy()

	for i in y['dscore'].index.values:
	  if (y['dscore'][i] > 0.65):
	    y['dscore'][i] = 1;
	  else:
	    y['dscore'][i] = 0;

	#do min max normalization
	for i in x:
		minimum = np.min(x[i])
		maximum = np.max(x[i])
		x[i] = min_max_normalization(x[i], minimum, maximum)

	#run cross validation with different number of k
	accuracy = []
	runtime = []
	for k in range(1, 10):
		print("k = %d" % (k*5))
		start_time = time.time()
		accuracy.append(np.mean(knn_cross_valid(10, x.shape[0], x.to_numpy(), y.to_numpy(), k*5)))
		print(accuracy[k-1])
		end_time = time.time()
		runtime.append(end_time - start_time)
		print(runtime[k-1])

	#plot accuracy
	plt.bar([5, 10, 15, 20, 25, 30, 35, 40, 45], accuracy)
	plt.title("K Nearest Neighbors")
	plt.ylabel("accuracy")
	plt.xlabel("k")
	plt.show()

	#plot runtime
	plt.bar([5, 10, 15, 20, 25, 30, 35, 40, 45], runtime)
	plt.title("K Nearest Neighbors")
	plt.ylabel("runtime")
	plt.xlabel("k")
	plt.show()

if __name__ == "__main__":
	main()

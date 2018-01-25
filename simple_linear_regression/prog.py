# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'




import tensorflow as tf
import csv
import math
import matplotlib.pyplot as plt

class mywrapper:
	pass

def readFromCSV_x(file):

	x = []
	with open(file, 'r') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
		for row in spamreader:
			x.append(math.floor(float(row[0])))

	return x

def readFromCSV_y(file):

	y = []
	with open(file, 'r') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
		for row in spamreader:
			y.append(math.floor(float(row[1])))

	return y

def getTrainingDataSet_x():

	#return [1.1, 2.1, 3.1, 4.2]
	return readFromCSV_x('./dataset/train.csv')

def getTrainingDataSet_y():

	#return [1.2, 2.1, 3.01, 4.21]
	return readFromCSV_y('./dataset/train.csv')

def getTestDataSet_x():

	return readFromCSV_x('./dataset/test.csv')

def getTestDataSet_y():

	return readFromCSV_y('./dataset/test.csv')

def pre_process_dataset(mywrapper):

	x = mywrapper.x
	y = mywrapper.y

	table1 = []
	table2 = []

	for i in range(0,len(x)):
		if x[i] in table1:
			continue
		else:
			table1.append(x[i])
			table2.append(y[i])

	x = table1
	y = table2

	mywrapper.x = x
	mywrapper.y = y

# Model inputs and outputs
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
x_train = getTrainingDataSet_x()
y_train = getTrainingDataSet_y()

mywrapper.x = x_train
mywrapper.y = y_train
pre_process_dataset(mywrapper)
x_train = mywrapper.x
y_train = mywrapper.y

x_test = getTestDataSet_x()
y_test = getTestDataSet_y()

mywrapper.x = x_test
mywrapper.y = y_test
pre_process_dataset(mywrapper)
x_test = mywrapper.x
y_test = mywrapper.y

#plt.plot(x_train, y_train, 'ro')
#plt.show()

# Model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
linear_model = W * x + b

# Loss function
loss = tf.reduce_sum(tf.square(linear_model - y))

# Initialize the weights
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Run the algo
optimizer = tf.train.GradientDescentOptimizer(0.00000001)
train = optimizer.minimize(loss)
for i in range(10000):
	sess.run(train, {x: x_train, y: y_train})

# Results
result = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: " + str(result[0]) + " b : " + str(result[1]) + " Loss: " + str(result[2]))


# Testing
print("Testing on test data")
cumulative = 0
runs = 0
for i in range(0,len(x_test)):

	if y_test[i] == 0:
		continue

	predicted_value = W.eval(sess) * x_test[i] + b.eval(sess)
	diff = ((y_test[i] - predicted_value) / y_test[i]) * 100
	cumulative += diff
	runs += 1
	#print("Actual value: " + str(y_test[i]) + " Predicted value: " + str(predicted_value) + " Percentage Difference: " + str(diff) + "%")

cumulative /= runs
print("Error rate: " + str(cumulative))

#File_Writer = tf.summary.FileWriter('/Users/adityach/Documents/dev/my_tensorflow', sess.graph)

sess.close()

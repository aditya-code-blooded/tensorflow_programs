# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



# Get task number from command line
import sys
import time
task_number = int(sys.argv[1])

if len(sys.argv) == 2:
	sleep_time = 0
elif len(sys.argv) == 3:
	sleep_time = int(sys.argv[2])
else:
	print("Error in passing command line arguments")
	exit(1)

import tensorflow as tf
import csv
import math

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

def getTrainingDataSet_x(num):

	#return [1.1, 2.1, 3.1, 4.2]
	file_name = './dataset/train' + str(num) + '.csv'
	print("Opening: " + file_name)
	return readFromCSV_x(file_name)

def getTrainingDataSet_y(num):

	#return [1.2, 2.1, 3.01, 4.21]
	file_name = './dataset/train' + str(num) + '.csv'
	print("Opening: " + file_name)
	return readFromCSV_y('./dataset/train' + str(num) + '.csv')

def getTestDataSet_x(num):

	file_name = './dataset/train' + str(num) + '.csv'
	print("Opening: " + file_name)
	return readFromCSV_x('./dataset/test' + str(num) + '.csv')

def getTestDataSet_y(num):

	file_name = './dataset/train' + str(num) + '.csv'
	print("Opening: " + file_name)
	return readFromCSV_y('./dataset/test' + str(num) + '.csv')

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


'''
	Add the cluster specification.
	In distributed tensorflow a cluster contains a set of jobs. Each job contains a set of tasks.
	Here we have 2 types of jobs: Worker and PS.

	Furthermore, we have 4 workers running on various ports as specified below. We also have a
	parameter task, also called parameter server running on another port.

'''
cluster = tf.train.ClusterSpec({
    "worker": [
        "localhost:2222",
        "localhost:2223",
        "localhost:2224"#,
        #"localhost:2225"
    ],
    "ps": [
        "localhost:2221"
    ]})

'''
	Suppose we run this file as:

		$python3 worker.py 2

	Then we will be creating a process which a worker with task number '2'
	and which accepts connections on port 2224

'''
server = tf.train.Server(cluster, job_name="worker", task_index=task_number)

print("Starting server #{}".format(task_number))

with tf.device("/job:ps/task:0"):
	
	# Model parameters
	W = tf.Variable([.3], dtype=tf.float32)
	b = tf.Variable([-.3], dtype=tf.float32)
	
with tf.device("/job:worker/task:" + str(task_number)):

	# Model inputs and outputs
	x = tf.placeholder(tf.float32)
	y = tf.placeholder(tf.float32)

	# The model
	linear_model = W * x + b


	# Read training data

	x_train = getTrainingDataSet_x(task_number)
	y_train = getTrainingDataSet_y(task_number)

	mywrapper.x = x_train
	mywrapper.y = y_train
	pre_process_dataset(mywrapper)
	x_train = mywrapper.x
	y_train = mywrapper.y

	print("Reading training data completed")

	'''

	# Read test data

	x_test = getTestDataSet_x(task_number)
	y_test = getTestDataSet_y(task_number)

	mywrapper.x = x_test
	mywrapper.y = y_test
	pre_process_dataset(mywrapper)
	x_test = mywrapper.x
	y_test = mywrapper.y

	'''

	# Sample plotting
	#plt.plot(x_train, y_train, 'ro')
	#plt.show()

	# Loss function

	loss = tf.reduce_sum(tf.square(linear_model - y))

	# Start the session

	print("Starting the session")

	sess = tf.Session(server.target)

	# Initialize the weights

	print("Initializing global variables")

	init = tf.global_variables_initializer()

	print("Running init")

	sess.run(init)

	# Run the algo

	optimizer = tf.train.GradientDescentOptimizer(0.00000001)
	train = optimizer.minimize(loss)

	# Wait for sometime and run
	# Check whether the W and b values are being updated or not
	time.sleep(sleep_time)

	result = sess.run([W, b, loss], {x: x_train, y: y_train})
	print("W: " + str(result[0]) + " b : " + str(result[1]) + " Loss: " + str(result[2]))

	for i in range(10000):

		if i % 1000 == 0:
			print("Running iteration: " + str(i))
		sess.run(train, {x: x_train, y: y_train})

	# Results

	result = sess.run([W, b, loss], {x: x_train, y: y_train})
	print("W: " + str(result[0]) + " b : " + str(result[1]) + " Loss: " + str(result[2]))


	'''# Testing

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
	print("Error rate: " + str(cumulative))'''

	#File_Writer = tf.summary.FileWriter('/Users/adityach/Documents/dev/my_tensorflow', sess.graph)

	sess.close()

print("Done!")
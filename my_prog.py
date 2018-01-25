# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

def getTrainingDataSet_x():

	return [1, 2, 3, 4]

def getTrainingDataSet_y():

	return [1, 2, 3, 4]

# Model inputs and outputs
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
x_train = getTrainingDataSet_x()
y_train = getTrainingDataSet_y()
#x_test = getTestDataSet_x()
#y_test = getTestDataSet_y()

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
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
for i in range(100):
	sess.run(train, {x: x_train, y: y_train})

result = sess.run([W, b, loss], {x: x_train, y: y_train})
print(result[0])

#File_Writer = tf.summary.FileWriter('/Users/adityach/Documents/dev/my_tensorflow', sess.graph)

sess.close()
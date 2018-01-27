# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



# Get task number from command line
import sys
task_number = int(sys.argv[1])

import tensorflow as tf

cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
server = tf.train.Server(cluster, job_name="local", task_index=task_number)

print("Starting server #{}".format(task_number))

server.start()
server.join()
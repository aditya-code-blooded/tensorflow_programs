# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



import tensorflow as tf

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

server = tf.train.Server(cluster, job_name="ps", task_index=0)

print("Starting parameter server #0")

server.start()
server.join()
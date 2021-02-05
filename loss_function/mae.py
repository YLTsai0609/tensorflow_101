import tensorflow as tf
import numpy as np

prediction = tf.constant([[1.0, 4.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 1.0]])
ground_truth = tf.constant([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [1.0, 1.0, 1.0]])


# 按照公式
error = tf.abs(tf.subtract(ground_truth, prediction))
mae = tf.reduce_mean(error)
with tf.Session() as sess:
    print(sess.run(mae))

# 套用API
mae1 = tf.losses.absolute_difference(ground_truth, prediction)
with tf.Session() as sess:
    print(sess.run(mae))

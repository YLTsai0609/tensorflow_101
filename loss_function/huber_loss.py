'''
more robust for outlier!

by downsampling the outlier.
'''
import tensorflow as tf
import numpy as np

prediction = tf.constant([[1.0, 4.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 1.0]])
ground_truth = tf.constant([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [1.0, 1.0, 1.0]])


# 參考：https://blog.csdn.net/hyk_1996/article/details/79570915
# 按照原理
def huber_loss(labels, predictions, delta=2.0):
    error = tf.abs(tf.subtract(ground_truth, prediction))
    condition = tf.less(error, delta)
    small_res = 0.5 * tf.square(error)  # 誤差小遵照L2
    large_res = delta * error - 0.5 * tf.square(delta)  # 誤差大採用線性誤差
    return tf.where(condition, small_res, large_res)


Huber = tf.reduce_mean(huber_loss(ground_truth, prediction, delta=2.0))
with tf.Session() as sess:
    print(sess.run(Huber))

# Call Tensorflow API
Huber_loss = tf.losses.huber_loss(ground_truth, prediction, delta=2.0)
with tf.Session() as sess:
    print(sess.run(Huber_loss))

import tensorflow as tf
import numpy as np

prediction = tf.constant([[1.0, 4.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 1.0]])
ground_truth = tf.constant([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [1.0, 1.0, 1.0]])

# 第一種方式，直接照著公式刻
Square_Error = tf.square(ground_truth - prediction)
mse = tf.reduce_mean(Square_Error)
with tf.Session() as sess:
    print(sess.run(mse))  # 1.8888888

# 透過L2 loss轉換
#  tf.nn.l2_loss=sum(error ** 2) / 2
mse1 = (2.0 * tf.nn.l2_loss(ground_truth - prediction)) / \
    tf.reshape(prediction, [-1]).get_shape().as_list()[0]
with tf.Session() as sess:
    print(sess.run(mse1))  # 1.8888888

# 直接套API
mse2 = tf.losses.mean_squared_error(ground_truth, prediction)
with tf.Session() as sess:
    print(sess.run(mse2))  # 1.8888888

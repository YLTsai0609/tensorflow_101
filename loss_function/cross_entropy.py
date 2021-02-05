import tensorflow as tf
import numpy as np

prediction = tf.constant([[1.0, 4.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 1.0]])
ground_truth = tf.constant(
    [[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=tf.float32)  # OneHotLabel
ground_truth_1 = tf.constant([1, 2, 0])  # index label

# 按照原理
log_sm_vals = tf.nn.log_softmax(prediction)
individual_loss = tf.reduce_sum(-1.0 *
                                tf.math.multiply(ground_truth, log_sm_vals), axis=1)
loss = tf.reduce_mean(individual_loss)
with tf.Session() as sess:
    print(sess.run(loss))  # 0.66168594

# API-1
cross_entropy_sum = tf.nn.softmax_cross_entropy_with_logits_v2(
    ground_truth, prediction)
cross_entropy = tf.reduce_mean(cross_entropy_sum)
with tf.Session() as sess:
    print(sess.run(cross_entropy))  # 0.66168594

# API-2 Sparse_Softmax，輸入不是OneHot
sparse = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=ground_truth_1, logits=prediction))
with tf.Session() as sess:
    print(sess.run(sparse))  # 0.66168594

# Binary Cross Entropy
sigmoid = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    labels=[0., 1., 0.], logits=[0.23, 1.23, 1.72]))

with tf.Session() as sess:
    print(sess.run(sigmoid))  # 0.98529524

# 與MSE的差異
mse = tf.losses.mean_squared_error(ground_truth, tf.nn.softmax(prediction))
with tf.Session() as sess:
    print(sess.run(mse))  # 0.13781698

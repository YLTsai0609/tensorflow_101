import tensorflow as tf
import numpy as np

tf.reset_default_graph()
# Label
label = tf.constant([0, 1, 0, 1, 0, 1, 0, 1], dtype=tf.int32)
# Embedding output
Embedding_out_out = tf.Variable(np.random.randn(8, 256), dtype=tf.float32)
# 經過L2-normalization
prediction_semi = tf.nn.l2_normalize(Embedding_out_out, axis=1)
# 計算 triplet loss
loss_semi = tf.contrib.losses.metric_learning.triplet_semihard_loss(
    label, prediction_semi)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(loss_semi))

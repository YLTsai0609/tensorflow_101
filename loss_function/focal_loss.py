import tensorflow as tf
import numpy as np

prediction = tf.constant([[1.0, 4.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 1.0]])
ground_truth = tf.constant(
    [[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=tf.float32)  # OneHotLabel
ground_truth_1 = tf.constant([1, 2, 0])  # index label


class FocalLoss():
    def __init__(self, gamma=2, alpha=[0.1, 0.2, 0.4], size_average=True):
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, prediction, ground_truth):

        softmax_prediction = tf.nn.softmax(prediction)
        softmax_prediction = tf.reduce_sum(
            softmax_prediction * ground_truth, 1)
        logpt = tf.log(softmax_prediction)

        if self.alpha is not None:
            at = tf.constant(self.alpha)

        loss = -1 * at * (1 - softmax_prediction)**self.gamma * logpt
        if self.size_average:
            return tf.reduce_mean(loss)
        else:
            return tf.reduce_sum(loss)


with tf.Session() as sess:
    Focal = FocalLoss()
    print(sess.run(Focal.forward(prediction, ground_truth)))  # 0.11024303


# 當gamma=0,alpha=1時，就是CE
with tf.Session() as sess:
    Focal = FocalLoss(gamma=0, alpha=[1., 1., 1.])
    print(sess.run(Focal.forward(prediction, ground_truth)))  # 0.661686


# 會與上方得到數值相同
cross_entropy_sum = tf.nn.softmax_cross_entropy_with_logits_v2(
    ground_truth, prediction)
cross_entropy = tf.reduce_mean(cross_entropy_sum)
with tf.Session() as sess:
    print(sess.run(cross_entropy))  # 0.661686

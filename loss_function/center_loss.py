import tensorflow as tf
import numpy as np

prediction = tf.constant([[1.0, 4.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 1.0]])
ground_truth = tf.constant(
    [[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=tf.float32)  # OneHotLabel
ground_truth_1 = tf.constant([1, 2, 0])  # index label

# 參考 https://github.com/EncodeTS/TensorFlow_Center_Loss


def get_center_loss(features, labels, alpha=0.1, num_classes=3):

    # Embedding的維度
    len_features = features.get_shape()[1]

    # 初始化center
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)

    centers_batch = tf.gather(centers, labels)

    # 計算Loss
    loss = tf.nn.l2_loss(features - centers_batch)

    # 更新各個類別的Center
    diff = centers_batch - features

    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff

    centers_update_op = tf.scatter_sub(centers, labels, diff)

    return loss, centers, centers_update_op


# 實際測試
tf.reset_default_graph()
Embedding = tf.Variable([[1, 2, ], [2, 3], [1, 2]], dtype=tf.float32)
loss, _, _ = get_center_loss(Embedding, [2, 0, 1], alpha=0.1, num_classes=3)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(loss))

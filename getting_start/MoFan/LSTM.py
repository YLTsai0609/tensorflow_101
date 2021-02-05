'''
我們這次還是做MNIST
但是用RNN做，這樣是基於什麼用途呢?
是這樣的
我們看1這張圖片
但是從第一行看，接下來看第2行
一直下去，看到最後一行，讓RNN記住這樣的序列關係
OK，我們開始吧

shape : 
input 28,
time_step = 28
X(128, 28, 28)
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import logging
logging.getLogger('tensorflow').disabled = True
tf.set_random_seed(1)   # set random seed

# 导入数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
lr = 0.001                  # learning rate
training_iters = 1600     # train step 上限
batch_size = 128            
display_step = 200

# 28*28，每次輸入28個pixel，跑28個time steps
n_inputs = 28               # MNIST data input (img shape: 28*28)
n_steps = 28                # time steps (每張圖總共要走28個steps)

n_hidden_units = 128        # neurons in hidden layer
n_classes = 10              # MNIST classes (0-9 digits)

# x y placeholder
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# 对 weights biases 初始值的定义
'''
資料要進入到RNN之前，會先進到一個hidden layer進行處理
RNN計算完之後，則會output到另一個hidden layer
hidden layer in : 28*128 fully connected layer
hidden layer out : 128*10 fully connected layer
'''
weights = {
    # shape (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # shape (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # shape (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # shape (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}

def RNN(X, weights, biases):
    # hidden layer for input to cell
    #######################################
    # X (128 batch, 28 steps, 28 inputs) (input是3dim)
    # ==> (128*28, 28) 一次 train 128張圖(?)
    with tf.name_scope('inputs'):
        X = tf.reshape(X, [-1, n_inputs], name='2_2D')
        # ==> (128*28, 128) 連接 fc - 1
        with tf.name_scope("Wx_plus_b"):
            X_in = tf.matmul(X, weights['in']) + biases['in']
        # ==> (128 batch, 28 steps, 128 hidden) (轉回 3 dim)
        X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
    #cell
    #######################################
    # 我們用128個RNN units訓練, 
    # forgetGate的初始bias定為1，因為一開始不希望忘記了前面的東西
    # state_is_turple (主線state值, 支線state值)
    # 我們的128個hidden units，每個都會計算出一個state，並加到目前的主線state，
    # 而tf中使用tuple來算，(主線,分線) -> (c_state, m_state)
    with tf.name_scope("LSTM"):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
        with tf.name_scope("initial_state"):
            _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
        # outputs -> list ： 會紀錄每一步的output，在我們的例子中，有128個rnn cell
        # states -> turple (c_state, m_state)
        # rnn vs dynamic rnn (dynamic rnn是一個比較好的形式)
        # time_major, 對於X_in, time_step是否在第一個維度? 在此例中在第2個維度，所以為false
        outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)

        # hidden layer for output as the final result
        #######################################
        # 取出分線劇情結果 states[1]進行加權
        with tf.name_scope("Wx_plus_b"):
            results = tf.matmul(states[1], weights['out']) + biases['out']

    return results



logits = RNN(x, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    writer = tf.summary.FileWriter("logs", sess.graph)
    # $ tensorboard --logdir='logs'

    # for step in range(1, training_iters+1):
    #     batch_x, batch_y = mnist.train.next_batch(batch_size)
    #     # Reshape data to get 28 seq of 28 elements
    #     batch_x = batch_x.reshape((batch_size, n_steps, n_inputs))
    #     # Run optimization op (backprop)
    #     sess.run(train_op, feed_dict={x: batch_x, y: batch_y})
    #     if step % display_step == 0 or step == 1:
    #         # Calculate batch loss and accuracy
    #         loss, acc = sess.run([loss_op, accuracy], feed_dict={x: batch_x,
    #                                                              y: batch_y})
    #         print("Step " + str(step) + ", Minibatch Loss= " + \
    #               "{:.4f}".format(loss) + ", Training Accuracy= " + \
    #               "{:.3f}".format(acc))

    # print("Optimization Finished!")

    # # Calculate accuracy for 128 mnist test images
    # test_len = 128
    # test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_inputs))
    # test_label = mnist.test.labels[:test_len]
    # print("Testing Accuracy:", \
    #     sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
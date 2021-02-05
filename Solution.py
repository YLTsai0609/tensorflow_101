# -*- coding: utf-8 -*-
# # Tensorflow like a sub-language built in python
# * Understanding how to use tensorflow and why the function works will give effiency on your work
# * Tensorflow vs Pytorch(2017-08-20)
#
# |比較項目|勝|Why|
# |-------|--|--|
# |上手時間|PyTorch|PyTorch本質上是Numpy的代替品，Tensorflow則是一個嵌入Python的程式語言|
# |圖創建與調試|PyTorch|PyTorch是動態圖，可以直接被python debugger使用，tensorflow不行，除了一些少見的動態結構(dynamic_rnn)|
# |全面性|Tensorflow|對於張量運算的支援還是比較多，例如檢察非數值張亮，快速傅立葉變換|
# |序列化|Tensorflow|tf可以將整個圖保存為protocol buffer，還能夠被其他語言加載(C++, Java)|
# |部署|Tensorflow|移動端以及嵌入式部署在Andoid/IOS只需要很少的工作量，甚至不用Java/C++|
# |文檔|PyTorch|tf的文件寫得很爛...|
# |設備管理|Tensorflow|tf通常預設你要使用GPU，只需要一行，PyTorch還挺麻煩的|
# [Reference](https://zhuanlan.zhihu.com/p/28636490)
#
#
#
# ## Leave Questions
# Why this
# 需求 : 需要使用tensorflow開發，要了解如何debugger
# 像是了解怎麼用print, 怎麼顯示出張量的shape等
#

import numpy as np 
import tensorflow as tf

'''
使用Session運算
TensorFlow Basic
tensorflow與其他數值運算package最大的不同是，
tensorflow是象徵性符號的(symbolic)
這使得tensorflow有一個大的優勢，可以做自動微分，這是numpy做不到的
同時也帶來一個缺點，就是比起numpy，tensorflow比較難掌握，
下面舉個例子與numpy對比
'''
x = np.random.normal(size=[3, 3])
y = np.random.normal(size=[3, 3])
z = np.dot(x, y)
print(z, type(z))
# tf
x = tf.random_normal([3,3])
y = tf.random_normal([3,3])
z = tf.matmul(x, y)
sess = tf.Session()
z_val = sess.run(z)
print(z_val, type(z_val))

# +
# '''
# 所以象徵符號式運算(symbolic computation)的優勢在哪裡?
# Assume that we have samples from a curve 
# (say f(x) = 5x^2 + 3)
# and we want to estimate f(x) based on these samples.
# We define a parametric function g(x, w) = w0 x^2 + w1 x + w2, 
# our goal is then to find the latent parameters such that g(x, w) ≈ f(x).
#  This can be done by minimizing the following loss function: L(w) = ∑ (f(x) - g(x, w))^2.
# Although there's a closed form solution for this simple problem, 
# 我們想要用g來逼近f，使用SGD
# [此段skip](https://github.com/vahidk/EffectiveTensorflow)
# '''
# x = tf.placeholder(tf.float32)
# y = tf.placeholder(tf.float32)
# x_data = np.random.normal(size=[3, 3])
# # print(x_data)
# with tf.Session() as s:
#     eval_x = s.run(x, {x : x_data})
#     print(x.shape, type(x)) # tf.placeholder -> unkown, tensor
#     print(eval_x.shape, type(eval_x), eval_x) # need to feed a value to x
# -

'''
靜態dimention與動態dimension
static and dynamic shapes
static shape : 再計算圖建構時決定(可能為不確定的)
dynamic shape : 
'''
a = tf.placeholder(tf.float32, [None, 128])
static_shape = a.shape.as_list() # return [None, 128], list

# get dynamic shape you can call tf.shape op
dynamic_shape = tf.shape(a) # Tensor("Shape:0", shape=(2,) dtype=int32)
print(static_shape)
print(dynamic_shape)

# static and dynamic shape could be set
a.set_shape([32, 128])
a.set_shape([None, 128])
a = tf.reshape(a, [32, 128])

# define a func return static shape if available
def get_shape(tensor:'tensor') -> list:
    static_shape = tensor.shape.as_list()
    dynamic_shape = tf.unstack(tf.shape(tensor))
    dims = [s[1] if s[0] is None else s[0]
           for s 
           in zip(static_shape, dynamic_shape)]
    return dims
b = tf.placeholder(tf.float32, [None, 10, 32])
shape = get_shape(b)

'''
tensorflow debugger
tf.assert
tf.assert系列東西滿多的，有rank, equal, rank_ast_least,....
tf.Print
'''
a = tf.random_uniform([2,3])
b = tf.random_uniform([3,4])
c = tf.matmul(a, b) # c is a tensor of shape[2, 4]


check_a = tf.assert_rank(a, 2) # if not, raise an InvalidArgementError exception
check_b = tf.assert_rank(b, 2)
with tf.control_dependencies([check_a, check_b]):
    c = tf.matmul(a, b)
    print(c.shape.as_list())


# +
'''
在function中確認張量shape
確認tensor有的method
關注82行-87行
'''

'''
tensorflow.__version__ = 1.8.0
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import logging
logging.getLogger('tensorflow').disabled = True



BATCH_START = 0
TIME_STEPS = 20
BATCH_SIZE = 50
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 10
LR = 0.006


def get_batch():
    global BATCH_START, TIME_STEPS
    # xs shape (50batch, 20steps)
    xs = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10*np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS
    # plt.plot(xs[0, :], res[0, :], 'r', xs[0, :], seq[0, :], 'b--')
    # plt.show()
    # returned seq, res and xs: shape (batch, step, input)
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]

class LSTMRNN(object):
    '''
    docstring
    '''
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size):
        self.n_steps = n_steps # 20
        self.input_size = input_size # 1
        self.output_size = output_size # 1
        self.cell_size = cell_size # 10
        self.batch_size = batch_size # 50
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
            self.ys = tf.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)

    def add_input_layer(self,):
        '''
        '''
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  # (batch*n_step, in_size)
        Ws_in = self._weight_variable([self.input_size, self.cell_size])
        bs_in = self._bias_variable([self.cell_size,])
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')

    def add_cell(self):
        
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)

        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)

    def add_output_layer(self):
        # 我想要在 self.cell_outputs reshape之前看一下他的shape，那我要怎麼做到??
        print(type(self.cell_outputs))
        print(dir(self.cell_outputs))
        print(self.cell_outputs.shape)
        print(self.cell_outputs._shape)
        print(self.cell_outputs.op)
        print(self.cell_outputs.graph)
        print(self.cell_outputs.name)
        print(type(self.cell_outputs.shape))
        l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')
        
        Ws_out = self._weight_variable([self.cell_size, self.output_size])
        bs_out = self._bias_variable([self.output_size, ])
        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.matmul(l_out_x, Ws_out) + bs_out

    def compute_cost(self):
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred, [-1], name='reshape_pred')],
            [tf.reshape(self.ys, [-1], name='reshape_target')],
            [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name='losses'
        )
        with tf.name_scope('average_cost'):
            self.cost = tf.div(
                tf.reduce_sum(losses, name='losses_sum'),
                self.batch_size,
                name='average_cost')
            tf.summary.scalar('cost', self.cost)

    @staticmethod
    def ms_error(labels, logits):
        return tf.square(tf.subtract(labels, logits))

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)


if __name__ == '__main__':
    # data for i in range(5):
    seq, res, xs = get_batch()
    # print(seq.shape, res.shape, xs.shape)
    # reset graph，不然code block不能執行，因為已經有這張計算圖了!
    tf.reset_default_graph()
    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", sess.graph)
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    # relocate to the local dir and run this line to view it on Chrome (http://0.0.0.0:6006/):
    # $ tensorboard --logdir='logs'

    # plt.ion()
    # plt.show()
    # for i in range(200):
    #     seq, res, xs = get_batch()
    #     if i == 0:
    #         feed_dict = {
    #                 model.xs: seq,
    #                 model.ys: res,
    #                 # create initial state
    #         }
    #     else:
    #         feed_dict = {
    #             model.xs: seq,
    #             model.ys: res,
    #             model.cell_init_state: state    # use last state as the initial state for this run
    #         }

    #     _, cost, state, pred = sess.run(
    #         [model.train_op, model.cost, model.cell_final_state, model.pred],
    #         feed_dict=feed_dict)

    #     # plotting
    #     plt.plot(xs[0, :], res[0].flatten(), 'r', xs[0, :], pred.flatten()[:TIME_STEPS], 'b--')
    #     plt.ylim((-1.2, 1.2))
    #     plt.draw()
    #     plt.pause(0.3)

    #     if i % 20 == 0:
    #         print('cost: ', round(cost, 4))
    #         result = sess.run(merged, feed_dict)
    #         writer.add_summary(result, i)

# -



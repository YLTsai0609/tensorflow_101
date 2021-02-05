import tensorflow as tf
import numpy as np
import logging
import matplotlib.pyplot as plt
logging.getLogger('tensorflow').disabled = True

BATCH_START = 0
TIME_STEPS = 20
BATCH_SIZE = 50
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 10
LR = 0.006
# [Note] No data size here 

def get_batch(time_steps : int,
             batch_size : int ) -> 'list[arr, arr, arr]':
    '''
    generate batch data
    TODO just the same operation about _check_array
    '''
    global BATCH_START
    basis = np.arange(BATCH_START, BATCH_START + time_steps * batch_size).reshape(batch_size, time_steps) / (10*np.pi)
    xs = np.sin(basis)
    ys = np.cos(basis)
    BATCH_START += time_steps
    result_list = [
                   xs[:, :, np.newaxis],
                   ys[:, :, np.newaxis],
                   basis
                 ]
    return result_list

class LSTM():
    '''
    batch = 50
    time_step = 20
    feature_dim = 1
    input:(50, 20, 1) -> reshape(1000, 1) -> fc1[1*10] -> (1000, 10) -> reshape(50, 20, 10)
    -> lstm2[10 cells] -> (50, 20, 10) -> reshape (1000, 10) -> fc3[10*1] -> prediction(1000, 1)
    '''
    def __init__(self, time_step, feature_dim, label_dim, cell_size, batch_size):
        self.time_step = time_step # 20
        self.feature_dim = feature_dim # 1
        self.label_dim = label_dim # 1
        self.cell_size = cell_size # 10
        self.batch_size = batch_size # 50
        self.weights = {
            'fc1':tf.Variable(tf.truncated_normal(shape=(self.feature_dim, self.cell_size),
                                                  stddev=0.1)),
            # 'lstm2':, tf.contrib.rnn.BasicLSTMCell, and tf.nn.dynamic_rnn
            'fc3': tf.Variable(tf.truncated_normal(shape=(self.cell_size, self.label_dim),
                                                  stddev=0.1)),
        }
        self.biases = {
            'fc1':tf.Variable(tf.zeros(shape=(self.cell_size))),
            # 'lstm2':tf.contrib.rnn.BasicLSTMCell, and tf.nn.dynamic_rnn
            'fc3':tf.Variable(tf.zeros(shape=(self.label_dim))),

        }
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, self.time_step, self.feature_dim], name='xs')
            self.ys = tf.placeholder(tf.float32, [None, self.time_step, self.label_dim], name='ys')
        with tf.name_scope('fc1'):
            fc1_in = tf.reshape(self.xs, [-1, self.feature_dim], name='fc1_in') # (50, 20, 1) -> (1000, 1)
            fc1_middle = self.get_dense_layer(fc1_in, self.weights['fc1'],self.biases['fc1'],
                                           activation=None)
            fc1 = tf.reshape(fc1_middle, [-1, self.time_step, self.cell_size]) #(50, 20, 1)
        with tf.name_scope('LSTM_cell'):
            lstm2 = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
            with tf.name_scope('initial_state'):
                '''
                共3項變數, cell_init_state, cell_outputs, cell_final_state
                '''
                self.lstm2_init_state = lstm2.zero_state(self.batch_size, dtype=tf.float32)
                lstm2_cell_outputs, self.lstm2_cell_final_state = tf.nn.dynamic_rnn(
                                            lstm2, fc1, initial_state=self.lstm2_init_state, time_major=False)
        with tf.name_scope('fc3'):
            fc3_in = tf.reshape(lstm2_cell_outputs, [-1, self.cell_size], name='fc3_in')
            self.pred = self.get_dense_layer(fc3_in, self.weights['fc3'],self.biases['fc3'],
                                activation=None)
        with tf.name_scope('loss'):
            self.compute_cost(self.pred, self.ys)
        with tf.name_scope('train'):
            pass
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)
        
    def get_dense_layer(self, input_layer, weight, bias, activation=None):
        '''
        全連接層
        '''
        x = tf.add(tf.matmul(input_layer, weight), bias) 
        if activation:
            x = activation(x)
        return x 
    def compute_cost(self, pred, label):
        '''
        legacy_seq2seq.sequence_loss_by_example 實作了 weighted cross entropy
        self.pred -> reshape : (1000,) every component is a tf.tensor
        self.ys -> reshape : (1000,) every component is a tf.tensor
        third component is weight of sequence component
        '''
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(pred, [-1], name='reshape_pred')],
            [tf.reshape(label, [-1], name='reshape_target')],
            [tf.ones([self.batch_size * self.time_step], dtype=tf.float32)],
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
        

# seq, res, xs = get_batch(TIME_STEPS, BATCH_SIZE)
# print(seq.shape, res.shape, xs.shape)

model = LSTM(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs", sess.graph)
print('Write graph into logs!')
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12

init = tf.global_variables_initializer()
sess.run(init)
for i in range(200):
    xs, ys, basis = get_batch(TIME_STEPS, BATCH_SIZE)
    if i == 0:
        feed_dict = {
                model.xs: xs,
                model.ys: ys,
                # create initial state
        }
    else:
        feed_dict = {
            model.xs: xs,
            model.ys: ys,
            model.lstm2_init_state: state    # use last state as the initial state for this run
        }
    
    _, cost, state, pred = sess.run(
            [model.train_op, model.cost, model.lstm2_cell_final_state, model.pred],
            feed_dict=feed_dict)

    # plotting
    plt.plot(basis[0, :], ys[0].flatten(),'r',label = 'target')
    plt.plot(basis[0, :], pred.flatten()[:TIME_STEPS], 'b--', label = 'predction')
    if i == 0:
        plt.legend(loc='upper right')
    plt.tight_layout()
    plt.ylim((-1.2, 1.2))
    plt.draw()
    plt.pause(0.3)

    if i % 20 == 0:
        print('cost: ', round(cost, 4))
        result = sess.run(merged, feed_dict)
        writer.add_summary(result, i)




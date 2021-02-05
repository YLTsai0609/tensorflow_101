'''
繼續講tensorboard
而且training step也可以被可視化
Graph可以看到靜態功能
########### Histogram ############
Histogram可以看到訓練過程
包含可以看到各層的
weight變化
bias變化
activation result變化
OFFSET模式:
以每個Time Step為切片，
histogram顯示layer中所有weight的分佈情況
########### Event(Scalar) #################
可以看到loss的變化，或是自己寫想要看的東西

########### Version #################
tensorflow 1.8.0
tf.summary.xxx
(1) tf.summary.scalar
(2) tf.summary.histogram
(3) tf.summary.merge_all
########### TensorBoard 解說 #################
https://blog.csdn.net/u010099080/article/details/77426577
'''


import tensorflow as tf
import numpy as np

# 想放在histogram看得就直接下一個histogram_summary
def add_layer(inputs, in_size, out_size,n_layer, activation_function=None):
    
    layer_name = 'n_layer%s' % n_layer
    with tf.name_scope(layer_name ):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name+'/weights', Weights)
            
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.summary.histogram(layer_name+'/biases', biases)
            
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
            tf.summary.histogram(layer_name+'/output', outputs)
        return outputs


# Make up some real data
# [:, np.new_axis] 其實就是reshape
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('inputs'):    
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# add hidden layer
l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1,n_layer=2, activation_function=None)

# 純量變化 tf.scalar_summary()
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                        reduction_indices=[1]))
    tf.summary.scalar('loss',loss)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()
# 把所有summary打包在一起
merged = tf.summary.merge_all()


# tf.train.SummaryWriter soon be deprecated, use following
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:  # tensorflow version < 0.12
    writer = tf.train.SummaryWriter('logs/', sess.graph)
else: # tensorflow version >= 0.12
    writer = tf.summary.FileWriter("logs/", sess.graph)

# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

# training step
for i in range(1000):
    sess.run(train_step, feed_dict ={xs:x_data, ys:y_data})
    if i%50 == 0:
        result = sess.run(merged,
                            feed_dict ={xs:x_data, ys:y_data})
        writer.add_summary(result, i)

# tensorboard --logdir path/to/log
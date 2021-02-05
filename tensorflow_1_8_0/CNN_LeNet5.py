# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import random
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
train_data = mnist.train
valid_data = mnist.validation
test_data = mnist.test

class CNNLogisticClassification:
    '''
    DocString here
    '''
    def __init__(self, shape_picture, n_labels, learning_rate=0.5,
                dropout_ratio=0.5,
                alpha=0):
        self.shape_picture = shape_picture
        self.n_labels = n_labels

        self.weights = None
        self.biases = None

        self.graph = tf.Graph() # 建構一張新的設計圖
        self.build(learning_rate, 
                  dropout_ratio, alpha) # 將神經網路網路參數加入設計圖
        self.sess = tf.Session(graph=self.graph) # 按照此設計圖開啟一個Session，準備訓練
    def build(self, learning_rate, dropout_ratio, alpha):
        with self.graph.as_default():
            # 輸入
            self.train_pictures = tf.placeholder(tf.float32,
                                                shape=[None]+self.shape_picture)
            self.train_labels = tf.placeholder(tf.float32, shape=(None, self.n_labels))

            ### 優化, y_ 即 y_hat : 模型預測的目標值
            self.y_, self.original_loss = self.structure(pictures=self.train_pictures, 
                                                labels=self.train_labels,
                                                dropout_ratio = dropout_ratio,
                                                train = True)

            # 損失函數正則項 Regularization 
            self.regularization = tf.reduce_sum(
                                  tf.reduce_mean([tf.nn.l2_loss(w) / tf.size(w, out_type=tf.float32)
                                  for w 
                                  in self.weights.values()]))
            
            # 損失函數 total loss 
            self.loss = self.original_loss + alpha * self.regularization

            # 定義優化操作張量
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            self.train_op = optimizer.minimize(self.loss)

            ### 預測
            self.new_pictures = tf.placeholder(tf.float32, shape = [None]+self.shape_picture)
            self.new_labels = tf.placeholder(tf.float32, shape = (None, self.n_labels))
            self.new_y_, self.new_original_loss = self.structure(pictures=self.new_pictures,
                                                                 labels=self.new_labels
                                                                )
            self.new_loss = self.new_original_loss + alpha * self.regularization 
            
            ### tensorflow變數初始化
            self.init_op = tf.global_variables_initializer()
    
    def structure(self, pictures, labels, dropout_ratio=0, train=False):
        '''
        LeNet5 Architecture(http://yann.lecun.com/exdb/lenet/)
        input:(batch, 28, 28, 1) -> conv1[5*5, 6] -> (batch, 24, 24, 6) (28-5+1 = 24)
        pool2(2*2) -> (batch, 12, 12, 6)  -> conv3[5*5*16] -> (batch, 8,8,16)
        pool4(2*2) -> (batch, 4,4,16) -> flatten5(batch, 256) -> fc6(batch, 120)
        -> fc7(barch, 84) -> fc8(batch, 10) -> softmax
        '''
        if (not self.weights) and (not self.biases):
            self.weights = {
                'conv1' : tf.Variable(tf.truncated_normal(shape=(5,5,1,6),
                                                        stddev=0.1)),
                'conv3' : tf.Variable(tf.truncated_normal(shape=(5,5,6,16),
                                                        stddev=0.1)),
                'fc6' : tf.Variable(tf.truncated_normal(shape=(4*4*16, 120),
                                                        stddev=0.1)),
                'fc7' : tf.Variable(tf.truncated_normal(shape=(120, 84),
                                                        stddev=0.1)),
                'fc8' : tf.Variable(tf.truncated_normal(shape=(84, self.n_labels),
                                                        stddev=0.1))
            }
            self.biases = {
                'conv1': tf.Variable(tf.zeros(shape=(6))),
                'conv3': tf.Variable(tf.zeros(shape=(16))),
                'fc6': tf.Variable(tf.zeros(shape=(120))),
                'fc7': tf.Variable(tf.zeros(shape=(84))),
                'fc8': tf.Variable(tf.zeros(shape=(10))),
            }
        conv1 = self.get_conv_2d_layer(pictures, self.weights['conv1'], self.biases['conv1'],
                                      activation=tf.nn.relu)
        pool2 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")
        conv3 = self.get_conv_2d_layer(pool2, self.weights['conv3'], self.biases['conv3'],
                                      activation=tf.nn.relu)
        pool4 = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")                                      
        flatten5 = self.get_flatten_layer(pool4)
        if train:
            flatten5 = tf.nn.dropout(flatten5, keep_prob = 1-dropout_ratio[0])
        fc6 = self.get_dense_layer(flatten5, self.weights['fc6'],self.biases['fc6'],
                                  activation=tf.nn.relu)
        if train:
            fc6 = tf.nn.dropout(fc6, keep_prob = 1-dropout_ratio[1])
        fc7 = self.get_dense_layer(fc6, self.weights['fc7'],self.biases['fc7'],
                                  activation=tf.nn.relu)
        logits = self.get_dense_layer(fc7, self.weights['fc8'],self.biases['fc8'])
        y_ = tf.nn.softmax(logits)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels = labels,
                                                    logits = logits)
        )
        return (y_, loss)
    def get_dense_layer(self, input_layer, weight, bias, activation=None):
        '''
        全連接層
        '''
        x = tf.add(tf.matmul(input_layer, weight), bias) # w_i * x_i + b
        if activation:
            x = activation(x)
        return x 
    def get_conv_2d_layer(self, input_layer, weight, bias, strides=(1,1), padding="VALID", activation=None):
        '''
        ConvolutionLayer2D
        VALID則會輸出較小shape的圖
        SAME則會輸出同樣shape的圖(其餘補0)
        '''
        x = tf.add(tf.nn.conv2d(input_layer,
                                weight,
                                [1, strides[0], strides[1], 1],
                                padding=padding), bias)
        if activation:
            x = activation(x)
        return x 
    def get_flatten_layer(self, input_layer):
        '''
        TODO why use n *= s? things
        '''
        shape = input_layer.get_shape().as_list()
        n = 1
        for s in shape[1:]:
            n *= s
        x = tf.reshape(input_layer, [-1, n])
        return x 
    def fit(self, X, y, epochs = 10, validation_data=None, test_data=None, batch_size=None):
        X = self._check_array(X)
        y = self._check_array(y)

        N = X.shape[0]
        random.seed(9000)

        if not batch_size:
            batch_size = N

        self.sess.run(self.init_op)
        for epoch in range(epochs):
            print('Epoch %2d/%2d' %(epoch+1, epochs))

            # Mini-Batch GD
            # batch_index使用min函數是為了最後一輪，當資料量不足一個batch時，全部選
            index = [i for i in range(N)]
            random.shuffle(index)
            while len(index) > 0:
                index_size = len(index)
                batch_index = [index.pop()
                               for _ 
                               in range(min(batch_size, index_size))]
            
                feed_dict = {self.train_pictures : X[batch_index, :], 
                            self.train_labels : y[batch_index],
                            }
            # TODO why 用兩個東西接?
                _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                
                print('[%d/%d loss = %9.4f      ' %(N-len(index), N, loss), end='\r')

            # 當前epoch計算指標分數
            y_ = self.predict(X)
            train_loss = self.evaluate(X, y)
            train_acc = self.accuracy(y_, y)
            msg = '[%d/%d] loss = %8.4f, acc = %3.2f%%' % (N, N, train_loss, train_acc*100)

            if validation_data:
                val_loss = self.evaluate(validation_data[0], validation_data[1])
                val_acc = self.accuracy(self.predict(validation_data[0]), validation_data[1])
                msg += '   val_loss = %8.4f, val_acc = %3.2f%%' %(val_loss, val_acc * 100)
            
            print(msg)

        if test_data:
            test_acc = self.accuracy(self.predict(test_data[0]), test_data[1])
            print('test_acc = %3.2f%%' %(test_acc * 100))
    def accuracy(self, predictions, labels):
        return (np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/predictions.shape[0])
    def predict(self, X):
        X = self._check_array(X)
        return self.sess.run(self.new_y_, feed_dict = {self.new_pictures : X})
    def evaluate(self, X, y):
        X = self._check_array(X)
        y = self._check_array(y)
        return self.sess.run(self.new_original_loss, feed_dict = {self.new_pictures : X, self.new_labels : y})
    def _check_array(self, ndarray):
        ndarray = np.array(ndarray)
        if len(ndarray.shape) == 1:
            ndarray = np.reshape(ndarray, (1, ndarray.shape[0]))
        return ndarray

model = CNNLogisticClassification(
    shape_picture=[28, 28, 1],
    n_labels=10,
    learning_rate=0.07,
    dropout_ratio=[0.2, 0.1],
    alpha=0.1,
)

train_img = np.reshape(train_data.images, [-1, 28, 28, 1])
valid_img = np.reshape(valid_data.images, [-1, 28, 28, 1])
test_img  = np.reshape(test_data.images, [-1, 28, 28, 1])

model.fit(
    X=train_img,
    y=train_data.labels,
    epochs=5,
    validation_data=(valid_img, valid_data.labels),
    test_data=(test_img, test_data.labels),
    batch_size=32,
)

# # Evaluaion
# * TODO 了解以下code怎麼work的

# 我們的物件中有什麼?
print(type(model))
attrAndMethods = [ele for ele in dir(model)
                      if not ele.startswith('__')]
print(attrAndMethods)
print()
print('-----Model Weights-------')
print()
print(model.weights)

# 第一層的Convolution Filters長什麼樣子?
# TODO 所以這裡有6個filter?
from matplotlib import pyplot as plt
# %matplotlib inline
fig, axis = plt.subplots(1,6, figsize=(7, 0.9))
for i in range(0,6):
    img = model.sess.run(model.weights['conv1'][:, :, :, i])
    img = np.reshape(img, (5,5))
    axis[i].imshow(img, cmap='gray')
plt.show()

# 第二層Convolution的Filters
fig, axis = plt.subplots(12,8, figsize=(10, 12))
for i in range(0,6):
    for j in range(16):
        img = model.sess.run(model.weights['conv3'][:, :, i, j])
        img = np.reshape(img, (5,5))
        axis[i*2+j//8][j%8].imshow(img, cmap='gray')
plt.show()


# +
# 我們試著丟幾張圖進去做Convolution，看一下在Filters的拆解之下圖片會變成怎樣?
# 第一行表示原圖
# 第二行表示做完第一次Convolution
# 第三行表示做完地二次Convolution
def showConvolutionFilter(model : 'CNNLogisticClassification',
                         image : 'np.array([x,y,colorDim])') -> None:
    fig, axis = plt.subplots(3, 16, figsize=(16,3))
    picture = np.reshape(image, (1,28,28,1))

    with model.sess.as_default():
        conv1 = model.get_conv_2d_layer(picture,
                                        model.weights['conv1'],
                                        model.biases['conv1'],
                                       activation=tf.nn.relu)
        pool2 = tf.nn.max_pool(conv1, 
                               ksize=[1,2,2,1],
                              strides=[1,2,2,1],
                               padding="VALID")

        conv3 = model.get_conv_2d_layer(pool2,
                                        model.weights['conv3'],
                                        model.biases['conv3'],
                                       activation=tf.nn.relu)
        eval_conv1 = conv1.eval()
        eval_conv3 = conv3.eval()

    axis[0][0].imshow(np.reshape(picture, (28,28)), cmap='gray')
    for i in range(6):
        img = eval_conv1[:, :, :, i]
        img = np.reshape(img, (24, 24))
        axis[1][i].imshow(img, cmap='gray')

    for i in range(16):
        img = eval_conv3[:, :, :, i]
        img = np.reshape(img, (8, 8))
        axis[2][i].imshow(img, cmap='gray')

showConvolutionFilter(model, test_img[0, :, :, :])
# -

showConvolutionFilter(model, test_img[10, :, :, :])

showConvolutionFilter(model, test_img[15, :, :, :])

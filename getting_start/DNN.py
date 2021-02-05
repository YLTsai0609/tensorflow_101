import numpy as np
import tensorflow as tf
import random
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
train_data = mnist.train
valid_data = mnist.validation
test_data = mnist.test

class DNNLogisticClassification:
    '''
    DNN - 結構為 Input_layer - Hidden_1_Dropout - softmax 
    cross-entropy(y_, y)
    Mini-Batch - 使用GD，但送入資料為random sampling
    random_seed在fit方法中寫死
    batch_size可在fit方法中設定參數
    '''
    def __init__(self, n_features : int, n_labels : int, learning_rate=0.5,
                n_hidden = 1000, activation=tf.nn.relu,dropout_ratio=0.5,
                alpha=0):
        self.n_features = n_features
        self.n_labels = n_labels

        self.weights = None
        self.biases = None

        self.graph = tf.Graph() # 建構一張新的設計圖
        self.build(learning_rate, n_hidden, activation,
                  dropout_ratio, alpha) # 將神經網路網路參數加入設計圖
        self.sess = tf.Session(graph=self.graph) # 按照此設計圖開啟一個Session，準備訓練

    def build(self, learning_rate, n_hidden, activation, dropout_ratio, alpha) -> None:
        '''
        根據神經網路參數建立設計圖
        其中對於本物件新增了
        train_features(經由tf.placeholder)
        train_labels(經由tf.placeholder)
        y_ (y_hat) (由本物件的strcuture函數傳回)
        loss (由本物件的strcuture函數傳回)
        train_op (優化操作張量)
        new_features (預測用特徵)
        new_labels (預測用標籤)
        new_y_y (預測的y_hat)
        init_op (tensorflow的初始化張量)
        '''
        with self.graph.as_default():
            # 輸入
            self.train_features = tf.placeholder(tf.float32, shape=(None, self.n_features))
            self.train_labels = tf.placeholder(tf.float32, shape=(None, self.n_labels))

            ### 優化, y_ 即 y_hat : 模型預測的目標值
            self.y_, self.original_loss = self.structure(features=self.train_features, 
                                                labels=self.train_labels,
                                                n_hidden = n_hidden,
                                                activation = activation,
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
            self.new_features = tf.placeholder(tf.float32, shape = (None, self.n_features))
            self.new_labels = tf.placeholder(tf.float32, shape = (None, self.n_labels))
            self.new_y_, self.new_original_loss = self.structure(features=self.new_features,
                                                                 labels=self.new_labels,
                                                                 n_hidden = n_hidden,
                                                                 activation = activation)
            self.new_loss = self.new_original_loss + alpha * self.regularization                                                                
                                                                 
                                                                 

            ### tensorflow變數初始化
            self.init_op = tf.global_variables_initializer()

    def structure(self, features, labels, n_hidden, activation, dropout_ratio=0, train=False):
        '''
        建立神經網路結構，return 預測值，loss
        '''
        ### Variables
        # weights與biases初始化，weight採取亂數, biases則為0
        # fc - fully-connected
        if (not self.weights) or (not self.biases):
            self.weights = {
                'fc1' : tf.Variable(tf.truncated_normal(shape=(self.n_features, n_hidden))),
                'fc2' : tf.Variable(tf.truncated_normal(shape=(n_hidden, self.n_labels))),
            }
            self.biases = {
                'fc1' : tf.Variable(tf.zeros(shape=(n_hidden))),
                'fc2' : tf.Variable(tf.zeros(shape=(self.n_labels))),
            }
        ### Struceture
        # Input Layer
        fc1 = self.get_dense_layer(features, 
                                   self.weights['fc1'],
                                   self.biases['fc1'],
                                   activation)
        if train:
            fc1 = tf.nn.dropout(fc1, keep_prob = 1-dropout_ratio)
        # layer2

        logits = self.get_dense_layer(fc1, self.weights['fc2'], self.biases['fc2'])

        # Predictions
        y_ = tf.nn.softmax(logits)
        # loss function : softmax cross entropy
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
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
            
                feed_dict = {self.train_features : X[batch_index, :], 
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
        return self.sess.run(self.new_y_, feed_dict = {self.new_features : X})
    def evaluate(self, X, y):
        X = self._check_array(X)
        y = self._check_array(y)
        return self.sess.run(self.new_original_loss, feed_dict = {self.new_features : X, self.new_labels : y})
    def _check_array(self, ndarray):
        ndarray = np.array(ndarray)
        if len(ndarray.shape) == 1:
            ndarray = np.reshape(ndarray, (1, ndarray.shape[0]))
        return ndarray
    
model = DNNLogisticClassification(
    n_features=28*28,
    n_labels=10,
    learning_rate=0.5,
    n_hidden=1000,
    activation=tf.nn.relu,
    dropout_ratio=0.5,
    alpha=0.01,
)
model.fit(
    X=train_data.images,
    y=train_data.labels,
    epochs=3,
    validation_data=(valid_data.images, valid_data.labels),
    test_data=(test_data.images, test_data.labels),
    batch_size = 32,
)
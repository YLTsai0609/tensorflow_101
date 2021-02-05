import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
train_data = mnist.train
valid_data = mnist.validation
test_data = mnist.test

'''
疑問點 
as_fefault()
reduce_mean()
accuracy()
TODO
type hinting
基本tensorflow張量運算
'''
class SimpleLogisticClassification:
    '''
    SimpleLogisticRegression，也就是沒有Hidden-Layer的單層NN
    class將會分成3個part
    建構(Building)，訓練(Fitting)，推論(inference)
    build : 在__init__中進行，由build建立Graph
            其中將Neuurel Network的結構分離存於 structure中
    fit : 訓練，這裡採用Gradient Descent
    predict : 根據訓練資料 X, 輸出模型預測值
    evaluate : 根據驗證指標評估模型好壞(這裡採用acc)
    '''
    def __init__(self, n_features : int, n_labels : int, learning_rate=0.5):
        self.n_features = n_features
        self.n_labels = n_labels

        self.weights = None
        self.biases = None

        self.graph = tf.Graph() # 建構一張新的設計圖
        self.build(learning_rate) # 將learning_rate加入設計圖
        self.sess = tf.Session(graph=self.graph) # 按照此設計圖開啟一個Session，準備訓練

    def build(self, learning_rate) -> None:
        '''
        根據learning_rate建立設計圖
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
            self.y_, self.loss = self.structure(features=self.train_features, labels=self.train_labels)

            # 定義優化操作張量
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)

            ### 預測
            self.new_features = tf.placeholder(tf.float32, shape = (None, self.n_features))
            self.new_labels = tf.placeholder(tf.float32, shape = (None, self.n_labels))
            self.new_y_, self.new_loss = self.structure(features=self.new_features, labels=self.new_labels)

            ### tensorflow變數初始化
            self.init_op = tf.global_variables_initializer()

    def structure(self, features, labels):
        '''
        建立神經網路結構，return 預測值，loss
        '''
        ### Variables
        # weights與biases初始化，weight採取亂數, biases則為0
        if (not self.weights) or (not self.biases):
            self.weights = {
                'fc1' : tf.Variable(tf.truncated_normal(shape=(self.n_features, self.n_labels)))
            }
            self.biases = {
                'fc1' : tf.Variable(tf.zeros(shape=(self.n_labels)))
            }
        ### Struceture
        logits = self.get_dense_layer(input_layer = features, weight=self.weights['fc1'], bias = self.biases['fc1'])
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
    def fit(self, X, y, epochs = 10, validation_data=None, test_data=None):
        X = self._check_array(X)
        y = self._check_array(y)

        self.sess.run(self.init_op)
        for epoch in range(epochs):
            print('Epoch %2d/%2d' %(epoch+1, epochs))

            # GradientDescent 全部資料
            feed_dict = {self.train_features : X, self.train_labels : y}
            self.sess.run(self.train_op, feed_dict=feed_dict)

            # 當前epoch計算指標分數
            y_ = self.predict(X)
            train_loss = self.evaluate(X, y)
            train_acc = self.accuracy(y_, y)
            msg = ' loss = %8.4f, acc = %3.2f%%' %(train_loss, train_acc * 100)

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
        return self.sess.run(self.new_loss, feed_dict = {self.new_features : X, self.new_labels : y})
    def _check_array(self, ndarray):
        ndarray = np.array(ndarray)
        if len(ndarray.shape) == 1:
            ndarray = np.reshape(ndarray, (1, ndarray.shape[0]))
        return ndarray
    
model = SimpleLogisticClassification(n_features = 28*28, n_labels=10, learning_rate=0.5)
model.fit(
    X=train_data.images,y=train_data.labels,epochs=10,
    validation_data=(valid_data.images, valid_data.labels),
    test_data=(test_data.images, test_data.labels)
)
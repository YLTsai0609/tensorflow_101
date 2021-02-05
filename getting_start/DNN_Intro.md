# Why deep ? 
單層網路在研究上已經被證明可以為任意的數學函數，為何要多層?
一層層的神經網路表示為特徵抽取(Feature Extraction)的過程，多層表示每一層只抽取壹些資訊
後面那層則會根據本層在抽取壹些資訊，也就是說
(1) 多層的網路架構可以抑制對雜訊的過度反應，這等同於一種Regularization
(2) 每一層中對於後一層新的neuron都做了線性相加，也就是Aggregation，這就像是把每個前面一層的neuron當作一個模型，然後拿來平均
降低Variance，這也是一種Regularization
總結來說，多層比單層更具有Regularization的效果，這是多層的好處
* 五種Deep learning 的Regularization
1. 結構設計 - 限縮神經元數量，加深網路層數等
2. 限制Weight的大小，像是 L2 Regularizer (Weight elimination) / L1 (Sparse)
3. Early Stopping
4. Dropout
5. Denoising AutoEncoder
'''

# Hidden layer
```
def structure(self, features, labels, n_hidden, activation, dropout_ratio=0, train=False):
        '''
        建立神經網路結構，return 預測值，loss
        '''
        ### Variables
        # weights與biases初始化，weight採取亂數, biases則為0
        # 其中fc表示為fully-connected
        if (not self.weights) or (not self.biases):
            self.weights = {
                'fc1' : tf.Variable(tf.truncated_normal(shape=(self.n_features, self.n_labels))),
                'fc2' : tf.Variable(tf.truncated_normal(shape=(n_hidden, self.n_labels)))
            }
            self.biases = {
                'fc1' : tf.Variable(tf.zeros(shape=(n_hidden))),
                'fc2' : tf.Variable(tf.zeros(shape=(self.n_labels)))
            }
        ### Struceture
        # 第一層 (layer1)
        fc1 = self.get_dense_layer(input_layer = features, weight=self.weights['fc1'], bias = self.biases['fc1'])

        if train:
            fc1 = tf.nn.dropout(fc1, keep_prob = 1 - dropout_ratio)
        
        # 第二層 (layer2)
        logits = self.get_dense_layer(input_layer = fc1, weight = self.weights['fc2'], bias = self.biases['fc2'])

        # Predictions
        y_ = tf.nn.softmax(logits)
        # loss function : softmax cross entropy
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        )
        return (y_, loss)
```
# Activation Function
<img src = './images/activation function (DNN_intro).png'></img>

* 想要output [-1, 1] --> tanh
* 想要output [0, 1] --> sigmoid
> tanh vs sigmoid 

|acti-func|pros|cons|
|---------|---|----|
|tanh|梯度變化大於sigmoid，訓練效率好，輸出mean = 0，不會傳遞bias，具有極值的Normolization|梯度消失|
|sigmoid|具有極值的Normolization|極大與極小都會變成平的，在input值大的時候，導致訊號磨損，梯度消失|
|relu|正的部分線性，克服梯度消失|對極值沒有做Normolization|
* 梯度消失 - 當acitivation function在極大與極小處梯度 -> 0，做Backpropagation時，訊號就會不斷被磨損，在前面幾層傳遞時，就已經被磨損為，此時更新的梯度是0
* relu的配套措施 - 由於tanh, sigmoid將輸出值限制在一個範圍內，所以像是一種Normolization，但relu沒有，因此relu經常會搭配Nomolization layer一起使用，或是使用一種新的Activation function SELU
  
# Mini-Batch GD

|Method|pros|cons|
|------|----|----|
|GD|更新穩定度佳|計算時間長|
|SGD|快，根據中央及限定理，會趨近於原本結果|有variance，容易掉進local min，不穩定|
|Mini-Batch SGD|平衡計算時間及更新穩定度，有可能比SGD快(如果有GPU)，收斂速度不輸GD||

* 如果使用GPU運算，依照GPU設計適當k值
* Mini-Batch收斂速度為何可以和GD差不多?
  原因在於更新的次數，Mini-Batch一次看的資料量比較少，所以一個Epoch可以更新參數好幾次

```
def fit(self, X, y, epochs = 10, validation_data=None, test_data=None):
        X = self._check_array(X)
        y = self._check_array(y)

        self.sess.run(self.init_op)
        for epoch in range(epochs):
            print('Epoch %2d/%2d' %(epoch+1, epochs))

            # Mini-batch GD
            index = [i for i in range(N)]
            random.shuffle(index)
            while len(index) > 0:
            index_size = len(index)
            batch_index = [index.pop() for _ in range(min(batch_size, index_size))]

            feed_dict = {self.train_features : X[batch_index, :], self.train_labels : y[batch_index]}

            _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)      
```
* 按照資料總數N列出可能的index, 做random.shuffle隨機抽樣，取出batch_size進行更新，直到所有index都用盡

# Regulzarization
## L2 Regularizer
```
regularization = tf.reduce_mean(
    [tf.nn_l2_loss(w) / tf.size(w, out_type=tf.float32)
     for w in self.weights.values()])

loss = original_loss + alpha * regularization
```
* 將loss對weights做平均，當調整神經元數量時，alphs就不用特別調整
## Dropout
先隨機關閉部分神經元，使用較少的神經元推論(Regularization)
推論時使用全部神經元(Aggregation Model)，train出各個sub NN然後組合起來
* details 權重的Normolize : 
假設今天要輸出的值有10個，這10個值都是1，但是因為Dropout，變成5個1，此時輸出的值變少了，會導致當輪更新時後面的neuron**低估更新量**
在這個例子中，我們應該要將輸出*2倍，維持與原本同等的輸出，也就是說

* 如果我們Dropout $r$ 倍的神經元，則權重就要乘上 $\frac {1}{r}$ 倍
  
```
with tf.Session() as sess:
    S = tf.constant([[1,1,1,1,1,1,1,1],
                     [3,3,3,3,3,3,3,3],
                     [5,5,5,5,5,5,5,5]]) # 3 * 8
    print('Original S = ')
    print(S.eval())
    S_drop = tf.nn.dropout(S, keep_prob=0.5)
    print('Dropout S = ')
    print(S_drop.eval())


Original S =
[[1. 1. 1. 1. 1. 1. 1. 1.]
 [3. 3. 3. 3. 3. 3. 3. 3.]
 [5. 5. 5. 5. 5. 5. 5. 5.]]
Dropout S =
[[ 2.  0.  0.  2.  0.  0.  2.  0.]
 [ 0.  0.  6.  0.  0.  6.  0.  6.]
 [10. 10. 10.  0. 10. 10. 10. 10.]]
```
tensorflow會自動幫我們乘上$\frac {1} {r}$，這是一件好事，另外一點，tensorflow的dropout是隨機的，所以比例會接近我們想要的，但是不會剛好
，而放入的方式就是加入`if train: ...`
則會告訴NN，如果是訓練過程，就使用dropout，如果是預測過程，就不使用

# optimizer
* 看李宏毅吧!
* [這裡也有](http://ruder.io/optimizing-gradient-descent/)

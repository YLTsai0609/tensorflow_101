import tensorflow as tf
import numpy as np
tf.logging.set_verbosity(tf.logging.ERROR)
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
'''
根據不同設計不同形式但合理的流程(flow)，在使用數據來訓練Model
這就是tensorflow命名的由來，tensor + flow

Graph 事先決定NN的結構，
決定NN要怎麼連接，
決定哪一些窗口是可以由外部來置放數據
哪一些變數可以被訓練
哪一些不可以被訓練
怎麼樣優化這個系統
等等等等等
Graph像是設計圖，沒辦法進行訓練
Session則是可以執行設計的結構
'''
# my_graph = tf.Graph()

'''
Session是一個開始，會將Graph的結構複製一份，準備開始訓練模型
(1) with寫法
(2) 物件式寫法
'''
# (1) with tf.Session(graph = my_graph) as sess:
# my_session = tf.Session(graph=my_graph)
# print(dir(my_session))

'''
Tensorflow變量介紹
常數張量 Model中不會改變的值
變數張量(1) 未知且待優化，會使用Initializer來設定初始值
變數張量(2) 想要一個變數張量，但又不希望他的值因為最佳化改變 - trainable = False
置放張量(Placeholder) 擔任輸入窗口的角色 (None, 1000) (未知Data數量, 1000個特徵值)
置放張量(Placeholder) 在Graph階段沒有數值，必須等到Session階段才將數值輸入進去
操作型張量(1) 不含有實際數值 作用1 : 最佳化 - GradientDescentOptimizer
操作型張量(2) 初始化操作，在變數張量時有用到initializer，這些initializer在Graph中還不具有數值
必須使用 tf.global_variables_initializer()來給值，所以一定要放進graph裏頭
* truncated - 縮減的，常態分佈，但是只取2個標準差內的值
'''
# tensor_constant = tf.constant([1,2,3,4,5,6,7], dtype=tf.int32)
# tensor_variable = tf.Variable(tf.truncated_normal(shape=(3,5)))
# tensor_placeholder = tf.placeholder(tf.float32, (None, 1000))
# # tensor_op = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)
# init_op = tf.global_variables_initializer()

'''
Session操作
[張量] 元素具有兩個面向 : 功能與數值
Graph階段 - 只具有功能
可以將設計圖pirnt出來
Session階段 - 具有功能與數值
可以使用eval() / run()來進行數值計算
'''
g1 = tf.Graph()
with g1.as_default():
    x = tf.constant(1)
    y = tf.constant(1)
    sol = tf.add(x, y)
with tf.Session(graph=g1) as sess:
    print(sol.eval())

'''
使用placerholder來做到 x + y
安裝正確版本的tensorflow
https://blog.csdn.net/wlwlomo/article/details/82806118
'''




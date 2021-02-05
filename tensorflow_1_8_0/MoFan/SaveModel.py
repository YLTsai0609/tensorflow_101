"""
2016年, 7月 
目前沒辦法保存整個Model，只能保存Variable
換句話說要把框架在生出來，把變數塞回去
建議開一個資料夾來放，因為單一個Model就會有3個相關檔案
一定要定義dtype和確認shape是否一致，Model才會一樣
"""
import tensorflow as tf
import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息  
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error   
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error

## Save to file

# W = tf.Variable([[1,2,3], [3,4,5]], dtype=tf.float32, name="Weights")
# b = tf.Variable([[1,2,3]], dtype=tf.float32, name="biases")

# init = tf.global_variables_initializer()

# saver = tf.train.Saver()

# with tf.Session() as sess:
#     sess.run(init)
#     save_path = saver.save(sess, "./example_saved_model/example.ckpt")
#     print('Save to path : ', save_path)

'''
網路框架還是需要重新定義，基本上只有儲存Variable而已
必須記住儲存時的dtype和shape
載入時不需要定義init
載入時的名字一定要和儲存時一模一樣才讀得到
'''
# load variable

# 空的框架
tf.reset_default_graph()
import numpy as np
W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name="Weights")
b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name="biases")

# not need init step

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "./example_saved_model/example.ckpt")
    print("weights:", sess.run(W))
    print("biases:", sess.run(b))

"""
1.8版本，模型儲存方式 (2018-06-15)
https://blog.csdn.net/autoliuweijie/article/details/80709499
"""
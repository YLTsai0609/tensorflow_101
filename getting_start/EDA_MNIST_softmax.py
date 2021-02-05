import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
print(tf.__version__)
tf.logging.set_verbosity(tf.logging.ERROR)

def summary(ndarr : 'np.array') -> None:
    '''
    numpy的summary函數
    包含numpy元素
    最小值，最大值，平均，標準差，獨立值
    '''
    print(ndarr)
    print('* shape: {}'.format(ndarr.shape))
    print('* min: {}'.format(np.min(ndarr)))
    print('* max: {}'.format(np.max(ndarr)))
    print('* avg: {}'.format(np.mean(ndarr)))
    print('* std: {}'.format(np.std(ndarr)))
    print('* unique: {}'.format(np.unique(ndarr)))


def plot_fatten_img(ndarr):
    img = ndarr.copy()
    img.shape = (28,28)
    plt.imshow(img, cmap='gray')
    plt.show()


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

train_data = mnist.train
valid_data = mnist.validation
test_data = mnist.test

# 所經過之處理 28*28 ->784, Pixel經過nomolize，並且亂數重排

# summary(train_data.images)

plot_fatten_img(train_data.images[10,:])
summary(train_data.labels)

def softmax(x : 'np.array, list'):
    '''
    input : [3.0, 1.0, 0.2]
    output : [0.83, 0.11, 0.05]
    這種score function會讓最大的值保留最大的比重，其餘的盡量小
    總和 = 1
    '''
    max_score = np.max(x, axis=0)
    x = x - max_score
    exp_s = np.exp(x)
    sum_exp_s = np.sum(np.exp(x), axis=0)
    softmax = exp_s / sum_exp_s
    return softmax
print(softmax(np.array([3.0, 1.0, 0.2])))

'''
使用LSTM實作文章產生器 : 我們希望LSTM可以不斷的根據前文來猜測
下一個字母(Letters)應該要下什麼，如此一來，LSTM就可以幫我腦補成一篇文章
output [a-z] 26個類別
input 序列性資料，我們必須決定多久更新一個part，也就是Unrolling Number
TODO
unrolling, 
gradient clipping
def my_lstm_cell
'''
import os
import random
import string
import zipfile
from urllib.request import urlretrieve
import time

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)


LETTER_SIZE = len(string.ascii_lowercase) + 1  # [a-z] + ' '
FIRST_LETTER_ASCII = ord(string.ascii_lowercase[0])

def maybe_download(url, filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urlretrieve(url, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified %s' % filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename


def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        name = f.namelist()[0]
        data = tf.compat.as_str(f.read(name))
    return data


def char2id(char):
    if char in string.ascii_lowercase:
        return ord(char) - FIRST_LETTER_ASCII + 1
    elif char == ' ':
        return 0
    else:
        print('Unexpected character: %s' % char)
        return 0


def id2char(dictid):
    if dictid > 0:
        return chr(dictid + FIRST_LETTER_ASCII - 1)
    else:
        return ' '


print('Downloading text8.zip')
filename = maybe_download('http://mattmahoney.net/dc/text8.zip', './text8.zip', 31344016)

print('=====')
text = read_data(filename)
print('Data size %d letters' % len(text))

print('=====')
valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)
print('Train Dataset: size:', train_size, 'letters,\n  first 64:', train_text[:64])
print('Validation Dataset: size:', valid_size, 'letters,\n  first 64:', valid_text[:64])


# 取到 batch data之後，我們來看看究竟要產生怎樣格式的資料
'''
'''
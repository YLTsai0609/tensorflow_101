import numpy as np

BATCH_START = 0
BATCH_SIZE = 50
TIME_STEPS = 20

def get_batch():
    '''
    return
    seq : (50, 20, 1)
    res : (50, 20, 1)
    xs : (50, 20)
    '''
    global BATCH_START, TIME_STEPS
    # xs shape (50batch, 20steps)
    xs = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10*np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS
    # plt.plot(xs[0, :], res[0, :], 'r', xs[0, :], seq[0, :], 'b--')
    # plt.show()
    # returned seq, res and xs: shape (batch, step, input)
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]

result = get_batch()
info = [(type(e), len(e), e.shape) for e in result]
print(info)

'''
一組sequence為一筆data，裡面可以有k個data point
這是一組sequence的X，通常稱為xs
[0.         0.03183099 0.06366198 0.09549297 0.12732395 0.15915494
 0.19098593 0.22281692 0.25464791 0.2864789  0.31830989 0.35014087
 0.38197186 0.41380285 0.44563384 0.47746483 0.50929582 0.54112681
 0.5729578  0.60478878]
 對應的y值也會有20個data point
 cos(xs), len = 20
 以上稱為一組sequence data
 而這樣的sequence data有50筆
 則batch size = 50
 在這組sequence data中xs沒有彼此overlap
'''
print(len(result[2]),result[2].shape, result[2][0], result[2][1], sep='\n')
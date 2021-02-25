# [SGLEE:20210225THU_183600] 최초 작성

import math
import tensorflow as tf

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

x = 0 # 입력
y = 1 # 출력
w = tf.random.normal([1], 0, 1)
b = tf.random.normal([1], 0, 1)

for i in range(2000):
    output = sigmoid(x * w + 1 * b)
    error = y - output
    w = w + x * 0.1 * error
    b = b + 1 * 0.1 * error
    
    if i % 100 == 99:
        print("[{0:4d}] error: {1:8f} output: {2:8f}".format(i, error, output))
        
# 값 예측
print('y: ', sigmoid(x * w + 1 * b))
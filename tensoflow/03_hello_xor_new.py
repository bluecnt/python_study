# [SGLEE:20210224WED_222700] 최초 작성

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

x = np.array([[1,1], [1,0], [0,1], [0,0]])
y = np.array([[0], [1], [1], [0]])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=2, activation='sigmoid', input_shape=(2,)),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

# mse: Mean Squared Error (평균 제곱 오차)
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1), loss='mse')

print('================================================================================')
print('XOR Test (new)')
print('================================================================================')

model.summary()
# [SGLEE:20210225THU_135200] 책(시작하세요! 텐서플로 2.0 프로그래밍)에는 2,000으로 되어 있음
history = model.fit(x, y, batch_size=1, epochs=5000)

print('predict(): \n', model.predict(x))

plt.plot(history.history['loss'])
plt.show()

# for weight in model.weights:
#     print(weight)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0

# Convert target labels to one-hot encoded format
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Dense layer 사용해서 MLP 모델 구축
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(units=5, activation='sigmoid'))
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

opt = tf.keras.optimizers.RMSprop(learning_rate=0.01)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# 콜백 설정: 조건에서 모델 조기 종료 설정
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                            min_delta=0.001, # 개선된 것으로 간주할 최소 변화량
                                            patience=1, # x번의 epoch동안 데이터 손실이 개선되지 않으면 그만(위반 허용 횟수)
                                            verbose=1, # 콜백이 동작하는 동안 로그를 출력할지 여부 0:출력(x), 1:출력(O)
                                            mode='auto') # auto는 자동으로 감지, min은 감소할때, max는 증가할때 개선된 것으로 판단

def scheduler(epoch, lr):
    if epoch % 2 == 0 and epoch:
        return 0.1*lr
    return lr
callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

ret = model.fit(x_train, y_train, epochs=100, batch_size=200,
                validation_split=0.2, verbose=2, callbacks=[callback])

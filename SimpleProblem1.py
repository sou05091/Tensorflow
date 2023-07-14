import tensorflow as tf

train_x = [1,2,3,4,5,6,7]
train_y = [3,5,7,9,11,13,15]

a = tf.Variable(0.1)
b = tf.Variable(0.1)

def lossfunction(a, b):
    # (예측값 - 실제값)**2
    # tf.square(예측1 - 실제1) + ...  + tf.square(예측i - 실제i) -> tf.keras.losses.mse(train_y, y)  
    # 예측값
    y = train_x * a + b 
    return tf.keras.losses.mse(train_y, y)


opt = tf.keras.optimizers.Adam(learning_rate=0.01)

for i in range(3000):
    opt.minimize(lambda:lossfunction(a,b), var_list=[a,b])
    print(a.numpy(), b.numpy())

# 1. 모델만들기
# 2. optimizer, 손실함수 정하기
# 3. 학습하기(경사하강으로 변수값 업데이트 하기)













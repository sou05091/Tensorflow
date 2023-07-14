import tensorflow as tf
height = [170, 180, 175, 160]
shoes = [260, 270, 265, 255]

# 신발 사이즈 예측
#y = ax + b (y는 신발 사이즈 x는 키)
a = tf.Variable(0.1)
b = tf.Variable(0.2)


def lossfunction():
    # 실제값 - 예측값
    return tf.square(260 - (170 * a + b))

# 경사하강법을 도와줌 (변수들 자동 업데이트)
opt = tf.keras.optimizers.Adam(learning_rate=0.1)

# varlist = 경사하강법 하면서 업데이트 할 값들
for i in range(300):
    opt.minimize(lossfunction, var_list=[a,b])
    print(a.numpy(),b.numpy())


import tensorflow as tf

# (고정된 값)숫자, 리스트 담는곳
tensor = tf.constant(3)

# 계산식 간단하게 할 수 있음
tensor1 = tf.constant([1,2,3])
tensor2 = tf.constant([6,7,8])

rs = tensor1 + tensor2
rs1 = tf.add(tensor1, tensor2)
#tf.subtract()
#tf.divide()
#tf.multiply()
#tf.matmul()

# tensor로 행렬 표현
tensor3 = tf.constant([1,2],
                      [3,4])

# 0이 담긴 tensor
# 2행 2열의 0으로 가득찬 행렬을 생성해줌
tensor4 = tf.zeros([2,2])

# 3행 2열의 행렬을 2개 생성
ex = tf.constant([2,2,3])

# 가중치 w값 저장
w = tf.Variable(1.0)
ex1 = w.numpy()

# w값 변경할려면
w.assign(2)

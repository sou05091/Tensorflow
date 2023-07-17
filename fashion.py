import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 데이터 불러오기
(trainX, trainY),(testX, testY) = tf.keras.datasets.fashion_mnist.load_data()
#print(testX[0])
#print(testX.shape)
#print(trainY)

# 0~255숫자를 0~1숫자로 미리 압축
# 결과가 좋게 나올수도 있고 처리 시간도 단축된다
trainX = trainX/255.0
testX = testX/255.0

# 4dim으로 만들어주기 위해 변경
# (trainX.shape[0],28,28) -> (trainX.shape[0],28,28,1) 이렇게 변경
trainX = trainX.reshape((trainX.shape[0],28,28,1))
testX = testX.reshape((testX.shape[0],28,28,1))

# 이미지 보는법
#plt.imshow(testX[1])
#plt.gray()
#plt.colorbar()
#plt.show()

class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle','boot']

model = tf.keras.Sequential([
    # 32개의 다른 feature(이미지)생성, (3,3) = kernel size, padding = same -> 커널 사용시 이미지가 작아지는걸 예방
    # relu함수를 사용하는 이유는 이미지 숫자에 -라는 값이 있을수 없기 때문에 사용 (이미지를 숫자로 0~255 사이)
    # Conv2D는 4차원의 데이터를 입력해주어야 한다. (60000,28,28,1)
    tf.keras.layers.Conv2D(32,(3,3), padding="same", activation="relu", input_shape = (28,28,1)),
    # (2,2) 는 pooling size
    tf.keras.layers.MaxPooling2D((2,2)),

    # relu = 음수의 값을 전부 0으로  만들어준다
    #tf.keras.layers.Dense(128, input_shape = (28,28), activation="relu"),
    # 행렬을 1차원으로 압축시켜준다. / 안좋은점 : 예측 모델의 응용력이 없어진다. 이전 사용한 가중치들이 상관없이 사용됨 / 같은 모양의 이미지 이더라도 조금만 달라지면 모른다.
    tf.keras.layers.Flatten(),
    # 해결책 : convolutional layer로 feature extraction 한다. 
    # 1. 이미지에서 중요한 정보를 추려서 복사본 20장을 만든다.
    # 2. 그곳엔 이미지의 주요한 feature, 특성이 담겨있다.
    # 3. 이걸로 학습한다. 
    # kernal 디자인 / Convolutional + Pooling layer사용시 특징 추출 + 특징을 가운데로 모아준다.

    tf.keras.layers.Dense(64, activation="relu"),

    # 확률 예측 문제이므로 마지막 레이어 노드수를 카테고리 갯수만큼
    # [0.2, 0.4, 0.6, 0.2, 0.1 ...] , softmax = 0과 1사이의 숫자 압축, 카테고리 예측 문제에 사용, 예측한 10개 확률을 전부 더하면 1이 나온다
    # sigmoid는 : binary예측 문제에 사용 ex) (대학원 합격 여부, 개or고양이), 마지막 노드 갯수는 1개 이어야 한다. 
    tf.keras.layers.Dense(10, activation="softmax"),
])

# 모델 아웃라인 출력하기
model.summary()

# sparse_categorical_crossentropy = 카테고리 예측사용에 사용
# 원 핫 인코딩이 되어있을때 categorical_crossentropy사용, 정수로 되어있을때 sparse_categorical_crossentropy사용 

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
# validation_data= epoch1회 끝날때 마다 채점해준다.
model.fit(trainX, trainY, validation_data=(testX, testY),epochs=5)

score = model.evaluate(testX, testY)
print(score)

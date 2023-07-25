# 대학원 합격 불합격 예측하기
import tensorflow as tf
import pandas as pd
import numpy as np

# 파일 읽어들이기
data = pd.read_csv('data/gpascore.csv')
#print(data.isnull().sum())

# na값 제거 모두 제거해줌
data = data.dropna()
# data.fillna(100) -> 빈칸에 100이라는 숫자를 채워 넣어줌
# min, max, count 등등 있음

# values 리스트로 데이터 담아줌
y_data = data['admit'].values
x_data = []

# 한 행씩 data.append
for i, rows in data.iterrows():
    x_data.append([rows['gre'], rows['gpa'], rows['rank']])

# 딥러닝 모델 만드는 법 -> Sequential 신경망 레이어들 쉽게 생성
model = tf.keras.models.Sequential([
    # 숫자 = 노드의 갯수 / tanh = activaationfunction
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(128, activation='tanh'),
    # 마지막 최종노드에 도달 (1개의 노드) / 마지막 레이어는 항상 예측결과를 뱉어야함 0~1사이의 숫자 -> sigmoid (0~1의 숫자로 만들어줌)
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

# 븐류 문제일 때 사용하기 좋은 lossfunction
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# epochs=10 -> 10번 학습 시킴
# 일반리스틀 numpy array로 변환해야 함
model.fit(np.array(x_data), np.array(y_data), epochs=1000)

# x -> 데이터 삽입시 list형식으로 ex)x = [[380,3.21,3] ...[] [] []]
# y -> [0,1,1,1,0,0 ... 1]


# 예측 하기 (predict)
predict =  model.predict([[750, 3.70, 3],[400, 2.2, 1]])
print(predict)


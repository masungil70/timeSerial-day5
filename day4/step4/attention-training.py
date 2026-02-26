import numpy as np
import pandas as pd
import koreanize_matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import  Layer, Input, LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import os
import joblib

# 0. GPU 및 혼합 정밀도 설정 (선택 사항)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 1. 데이터 로드 및 전처리
data_df = pd.read_csv('./data/flights.csv')
passengers = data_df['Passengers'].values.astype(float)

# 계절성 차분 (Seasonal Differencing: 현재 - 12개월 전)
seasonal_period = 12
diff_passengers = passengers[seasonal_period:] - passengers[:-seasonal_period]
diff_passengers = diff_passengers.reshape(-1, 1)

# 데이터 정규화
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(diff_passengers)

# 모델/스케일러 저장 경로 확인
save_dir = './model'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 스케일러 저장
joblib.dump(scaler, os.path.join(save_dir, "air_passengers_scaler.pkl"))

# 시퀀스 생성 함수
def create_dataset(dataset, look_back=12):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)

look_back = 12
X, y = create_dataset(data_scaled, look_back)
X = X.reshape((X.shape[0], X.shape[1], 1))

# 학습/테스트 데이터 분리
train_size = len(X) - 24
X_train, X_test = X[:train_size], X[train_size:] # 또는 명시적으로 인덱싱
y_train, y_test = y[:train_size], y[train_size:]

# 2. Attention 레이어 정의 (Serialization 대응 최적화)
@tf.keras.utils.register_keras_serializable() # 모델 저장 및 로드 시 커스텀 레이어 인식을 위한 데코레이터
class AttentionLayer(Layer):
    """
    LSTM의 출력 시퀀스에서 중요한 정보에 집중(Attention)하여 
    가중 합산된 컨텍스트 벡터를 생성하는 커스텀 레이어입니다.
    """
    def __init__(self, **kwargs):
        # 부모 클래스(layers.Layer)의 초기화 루틴을 수행합니다.
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        레이어가 처음 호출될 때 실행되며, 학습 가능한 가중치(Weight)를 생성합니다.
        input_shape:     (Batch_size, Time_steps, Input_dim) 형태입니다.
        배열을 왼쪽에 접근   :  0           1           2
        배열을 왼른쪽에 접근 : -3          -2          -1
        """
        # 1. 학습 가능한 가중치 W 정의: 각 시점의 특징값에 곱해질 가중치 행렬
        # 형태: (입력 차원, 1) -> 각 시점의 벡터를 스칼라 점수로 변환하기 위함
        self.W = self.add_weight(name="att_weight", 
                                 shape=(input_shape[-1], 1), 
                                 initializer="normal",
                                 trainable=True)
        
        # 2. 편향 b 정의: 활성화 함수 적용 전 더해지는 학습 가능한 상수
        # 형태: (타임스텝 수, 1) -> 각 시점별로 고유한 편향값 부여
        self.b = self.add_weight(name="att_bias", 
                                 shape=(input_shape[1], 1), 
                                 initializer="zeros",
                                 trainable=True)
        
        # 가중치 생성이 완료되었음을 선언합니다.
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        """
        실제 연산이 일어나는 핵심 메서드 (Forward Propagation)
        inputs: LSTM의 출력값 (Batch, Time_steps, Feature_dim)
        """
        # [단계 1] 점수 계산 (Score Calculation)
        # inputs(W) + b 를 통해 각 시점의 중요도를 나타내는 '에너지 점수'를 계산합니다.
        # tanh 활성화 함수를 사용하여 점수를 -1과 1 사이로 비선형 변환합니다.
        et = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)

        # [단계 2] 확률 변환 (Attention Weights)
        # Softmax를 사용하여 모든 시점의 et 합계가 1(100%)이 되도록 확률값으로 변환합니다.
        # axis=1은 타임스텝 방향으로 소프트맥스를 적용한다는 의미입니다.
        at = tf.nn.softmax(et, axis=1)

        # [단계 3] 가중치 적용 (Weight Application)
        # 원본 입력값(inputs)에 계산된 확률 가중치(at)를 곱합니다.
        # 중요한 시점의 데이터는 크게 남고, 불필요한 시점은 0에 가깝게 작아집니다.
        context = inputs * at

        # [단계 4] 정보 합산 (Context Vector)
        # 가중치가 곱해진 모든 시점의 벡터를 하나로 합칩니다(Sum).
        # 결과값은 (Batch, Feature_dim) 형태의 '문맥 벡터'가 됩니다.
        # 가중치(at)도 나중에 시각화하기 위해 함께 반환합니다.
        return tf.reduce_sum(context, axis=1), at

    def get_config(self):
        """
        레이어의 설정 정보를 딕셔너리 형태로 반환합니다.
        이 함수가 있어야 model.save()로 저장된 모델을 나중에 완벽히 불러올 수 있습니다.
        """
        config = super(AttentionLayer, self).get_config()
        # 추가적인 하이퍼파라미터가 있다면 여기에 업데이트합니다.
        return config
    
# 3. 모델 구축 (Functional API 스타일 : Sequential보다 유연한 모델 정의 방식입니다)
inputs = Input(shape=(look_back, 1))
# LSTM의 모든 시점 출력을 위해 return_sequences=True
#LSTM 객체 생성 후 입력값을 전달하여 lstm_out에 저장
lstm_out = LSTM(128, return_sequences=True)(inputs) 
# Attention 적용
# AttentionLayer 객체 생성 후 lstm_out을 입력값으로 전달하여 attention_out과 attention_weights에 저장
attention_out, attention_weights = AttentionLayer()(lstm_out)

# 최종 출력
# Dense 레이어를 사용하여 attention_out에서 최종 예측값을 생성합니다.
prediction = Dense(1)(attention_out)

model = Model(inputs=inputs, outputs=prediction)
model.compile(optimizer='adam', loss='mse')

# 학습
print("🚀 Attention-LSTM 모델 학습 중...")
model.fit(X_train, y_train, epochs=300, batch_size=16, verbose=0)

# 모델 저장 (.h5 대신 최신 .keras 포맷 권장하나 사용자 설정에 맞춰 .h5 유지)
model.save(os.path.join(save_dir, "air_passengers_best_model.h5"))

# 4. 성능 검증 및 역변환
y_pred_diff_scaled = model.predict(X_test)
y_pred_diff = scaler.inverse_transform(y_pred_diff_scaled).flatten()

# 차분 데이터 복원 (이전 주기 값 + 차분 예측값)
actual_start_idx = len(passengers) - 24
y_pred_final = []
for i in range(24):
    prev_year_val = passengers[actual_start_idx - seasonal_period + i]
    y_pred_final.append(prev_year_val + y_pred_diff[i])

y_pred_final = np.array(y_pred_final)
y_actual_final = passengers[actual_start_idx:]

# MAPE 계산
mape = np.mean(np.abs((y_actual_final - y_pred_final) / y_actual_final)) * 100
print(f"📊 최종 모델 MAPE: {mape:.2f}%")

# 5. 결과 시각화
plt.figure(figsize=(12, 5))
plt.plot(y_actual_final, label='실제값', marker='o', alpha=0.7)
plt.plot(y_pred_final, label=f'예측값 (MAPE: {mape:.2f}%)', marker='x', color='red')
plt.title('항공기 승객 수 예측 (Attention-LSTM)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
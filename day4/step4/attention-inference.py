import numpy as np
import pandas as pd
import koreanize_matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import  Layer, Input, LSTM, Dense
import joblib
import os

# 1. Attention 레이어 클래스 정의 (로드 시 필수)
# @tf.keras.utils.register_keras_serializable()는 학습 시 등록된 이름을 찾기 위해 필요할 수 있습니다.
@tf.keras.utils.register_keras_serializable(package="Custom")
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


# 2. 파일 경로 설정
model_path = './model/air_passengers_best_model.h5'
scaler_path = './model/air_passengers_scaler.pkl'
data_path = './data/flights.csv'

# 파일 존재 확인
if not all(os.path.exists(f) for f in [model_path, scaler_path, data_path]):
    print("❌ 필요한 파일(모델, 스케일러, 데이터)이 없습니다. 경로를 확인해주세요.")
else:
    # 3. 모델 및 스케일러 로드
    # 'Custom>AttentionLayer' 에러를 방지하기 위해 custom_object_scope를 사용합니다.
    custom_objects = {'AttentionLayer': AttentionLayer}
    
    with tf.keras.utils.custom_object_scope(custom_objects):
        model = tf.keras.models.load_model(model_path, compile=False)
    
    scaler = joblib.load(scaler_path)
    print("✅ 모델 및 스케일러 로드 완료")

    # 4. 원본 데이터 로드 (미래 예측의 기준점)
    data_df = pd.read_csv(data_path)
    col_name = 'Passengers'
    passengers = data_df[col_name].values.astype(float)

    # 5. 미래 예측 (1961년 12개월)
    # 마지막 12개월의 차분 데이터 준비
    seasonal_period = 12
    diff_passengers = passengers[seasonal_period:] - passengers[:-seasonal_period]
    diff_scaled = scaler.transform(diff_passengers.reshape(-1, 1))
    
    # 마지막 시퀀스 (1960년 패턴)
    # Keras의 LSTM 입력 규격인 **(Samples, Time_steps, Features)**에 맞춰진 구조로 준비합니다.
    current_batch = diff_scaled[-12:].reshape(1, 12, 1)
    
    future_diff_preds = []
    print("🔮 1961년 미래 예측 진행 중...")
    
    for i in range(12):
        pred_scaled = model.predict(current_batch, verbose=0)
        future_diff_preds.append(pred_scaled[0, 0])
        
        # 윈도우 슬라이딩 업데이트
        new_val = pred_scaled.reshape(1, 1, 1)
        current_batch = np.append(current_batch[:, 1:, :], new_val, axis=1)

    # 6. 역변환 및 복원
    future_diff_unscaled = scaler.inverse_transform(np.array(future_diff_preds).reshape(-1, 1)).flatten()
    
    # 1961년 최종값 = 1960년 실제값 + 예측된 증감량
    last_year_1960 = passengers[-12:]
    forecast_1961 = last_year_1960 + future_diff_unscaled

    # 7. 시각화 및 출력
    future_months = pd.date_range(start='1961-01-01', periods=12, freq='MS')
    forecast_series = pd.Series(forecast_1961, index=future_months)

    plt.figure(figsize=(12, 6))
    plt.plot(pd.to_datetime(data_df['Month'])[-24:], passengers[-24:], label='실제값 (1959-1960)', marker='o')
    plt.plot(forecast_series, label='예측값 (1961)', marker='x', color='red', linestyle='--')
    plt.title('항공기 승객 수 미래 예측 (1961년)')
    plt.ylabel('승객 수')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    print("\n--- 1961년 예측 결과 ---")
    for month, val in zip(future_months, forecast_1961):
        print(f"{month.strftime('%Y-%m')}: {int(val)}명")
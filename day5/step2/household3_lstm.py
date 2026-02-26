import pandas as pd
import numpy as np
import koreanize_matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------------------------------------
# [단계 0] 하드웨어 및 환경 설정
# ---------------------------------------------------------

# 1. GPU 메모리 동적 할당 (RTX 30 시리즈 필수)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # 2. 혼합 정밀도(Mixed Precision) 설정: FP16 가속 활성화
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print(f"✅ 하드웨어 가속 활성화: {policy.name}")
    except RuntimeError as e:
        print(f"❌ 설정 오류: {e}")


# [1] 데이터 로드 및 전처리
# - 이전 단계와 동일하게 에너지 사용량과 날씨 데이터를 결합하고 주말 변수를 추가합니다.
df_energy = pd.read_csv(r'.\data\household_daily_usage.csv', parse_dates=['dt'], index_col='dt')
df_weather = pd.read_csv(r'.\data\paris_weather_data.csv', parse_dates=['time'], index_col='time')

df_weather['temp_est'] = (df_weather['tmin'] + df_weather['tmax']) / 2
df_weather = df_weather[['temp_est', 'tmin', 'tmax', 'prcp']].interpolate().ffill().bfill()
df = df_energy.join(df_weather, how='inner')
df['is_weekend'] = df.index.dayofweek.map(lambda x: 1 if x >= 5 else 0)

# [2] 학습 특성(Features) 선정
# - 타겟인 'Global_active_power'를 첫 번째 컬럼으로 두어, 이후 역스케일링(원래 값 복원) 시 편리하게 구성합니다.
features = ['Global_active_power', 'temp_est', 'tmin', 'tmax', 'prcp', 'is_weekend']
dataset = df[features].values

# [3] 데이터 정규화 (Scaling)
# - LSTM은 0과 1 사이의 값으로 정규화되었을 때 수렴 속도와 예측 성능이 가장 좋습니다.
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# [4] 시계열 시퀀스 데이터 생성 함수 (Window Sliding Technique)
# - 과거 일주일(seq_length)의 데이터를 보고 다음 날의 값을 예측하는 형태로 데이터셋을 변형합니다.
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        # i부터 i+seq_length까지의 6개 변수 데이터를 입력(X)으로 사용
        X.append(data[i:i+seq_length, :]) 
        # i+seq_length 시점의 첫 번째 컬럼(전력 사용량)을 정답(y)으로 사용
        y.append(data[i+seq_length, 0])    
    return np.array(X), np.array(y)

seq_length = 7 # 과거 7일치를 학습하여 8일째를 예측
X, y = create_sequences(scaled_data, seq_length)

# [5] 학습/테스트 데이터셋 분리
# - 순서가 중요한 시계열 데이터이므로 랜덤 셔플링 없이 시간 순서대로 마지막 30일을 테스트셋으로 설정합니다.
train_size = len(X) - 30
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# [6] LSTM 신경망 모델 설계
# - LSTM 계층: 시계열의 장단기 기억을 담당
# - Dropout: 과적합(Overfitting)을 방지하기 위해 무작위로 일부 뉴런을 끔
# - Dense(1): 최종적으로 다음 날의 전력량 1개를 예측
model = Sequential([
    # 입력 규격 정의: (7일의 타임스텝, 6개의 특성)
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    
    # 첫 번째 LSTM 계층: return_sequences=True는 다음 LSTM 층으로 시퀀스를 전달하기 위함
    LSTM(128, return_sequences=True),

    # 두 번째 LSTM 계층: return_sequences=False는 시퀀스를 요약하여 하나의 벡터로 만듦
    LSTM(64, return_sequences=False),
    
    # 출력 계층: 다음 날의 전력 사용량 1개 예측
    Dense(1) 
])

# [7] 모델 컴파일 및 학습
model.compile(optimizer='adam', loss='mse')

# EarlyStopping 설정: 최적의 가중치를 복구하도록 restore_best_weights 추가 권장
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True
)

print("🚀 다변량 LSTM 학습 시작...")
history = model.fit(
    X_train, y_train, 
    epochs=50, 
    batch_size=16, 
    validation_split=0.1, 
    callbacks=[early_stop], 
    verbose=1
)
# [8] 예측 및 역정규화 (Inversing Scaling)
# - 모델은 0~1 사이 값을 출력하므로, 이를 실제 단위인 kW로 복구해야 합니다.
predictions = model.predict(X_test)

# - 역스케일링을 수행하려면 학습 당시 사용한 6개 컬럼의 형식을 맞춰야 함 (더미 행렬 활용)
predict_copies = np.zeros((len(predictions), len(features)))
predict_copies[:, 0] = predictions.flatten() # 첫 번째 컬럼에 예측값 배치
inv_predictions = scaler.inverse_transform(predict_copies)[:, 0] # 원래 단위로 복원

# - 실제 값(y_test)도 비교를 위해 동일하게 역스케일링 진행
actual_copies = np.zeros((len(y_test), len(features)))
actual_copies[:, 0] = y_test
inv_actual = scaler.inverse_transform(actual_copies)[:, 0]

# [9] 최종 결과 시각화 및 비교
plt.figure(figsize=(12, 6))
plt.plot(df.index[-30:], inv_actual, label='Actual (실제)', color='blue', marker='o')
plt.plot(df.index[-30:], inv_predictions, label='LSTM (추론)', color='orange', linestyle='--', marker='s')
plt.title('에너지 소비량 추론 (LSTM)')
plt.ylabel('Global Active Power (kW)')
plt.legend()
plt.grid(True)
plt.show()
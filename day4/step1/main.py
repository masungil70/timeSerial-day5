import pandas as pd
import numpy as np
import koreanize_matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. 시각화 및 환경 설정
plt.rcParams['axes.unicode_minus'] = False 

# [단계 1] 데이터 로드 및 시간 변환
df = pd.read_csv('./data/power_usage_dataset.csv')
df['Date'] = pd.to_datetime(df['Date'])

# [단계 2] 특성 공학 (Cycle Encoding)
# 23시와 0시가 연속적임을 모델에게 알려주는 중요한 과정입니다.
df['hour'] = df['Date'].dt.hour
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 23)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 23)

features_list = ['Temperature', 'Usage', 'hour_sin', 'hour_cos']
data = df[features_list].values

# [단계 3] 데이터 전처리 (스케일링 및 시퀀스 생성)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

def create_sequences(data, window_size=24):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size, :]) 
        y.append(data[i + window_size, 1]) # Target: Usage
    return np.array(X), np.array(y)

window_size = 24 # 1주일(24시간) 패턴 학습
X, y = create_sequences(scaled_data, window_size)

# 데이터 분할
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"✅ 학습 데이터 규격: {X_train.shape} (샘플 수, 타임스텝, 피처 수)")

# [단계 4] 모델 설계 (최신 권장 스타일: Input 레이어 및 Stacked 구조)
model = Sequential([
    Input(shape=(window_size, 4)), # 입력 규격 명시
#    Input(shape=(X_train.shape[1], X_train.shape[2])), # X_train 변수를 활용한 규격 명시 (24, 4)로 변경
    LSTM(units=128, activation='tanh'),
    Dense(32, activation='tanh'),
    Dense(units=1)
])

# [단계 5] 컴파일 및 조기 종료 설정
model.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(
    monitor='val_loss',         # 감시 대상: 검증 데이터의 손실 값
    patience=7,                 # 성능 개선이 없을 때 기다려줄 에포크 횟수
    restore_best_weights=True   # 학습 종료 후 가장 성적이 좋았던 시점의 가중치로 복원
)

# 모델 학습
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop], #조기 종료 콜백 추가
    verbose=1
)

# [단계 6] 예측 및 역스케일링
predictions_scaled = model.predict(X_test)

def get_original_units(scaled_values, scaler, feature_count, target_idx=1):
    dummy = np.zeros((len(scaled_values), feature_count))
    dummy[:, target_idx] = scaled_values.flatten()
    return scaler.inverse_transform(dummy)[:, target_idx]

y_test_original = get_original_units(y_test, scaler, len(features_list))
predictions_original = get_original_units(predictions_scaled, scaler, len(features_list))

# [단계 7] 결과 시각화
plt.figure(figsize=(14, 6))
plt.plot(y_test_original[:168], label='실제값', color='#1f77b4', alpha=0.8, linewidth=2)
plt.plot(predictions_original[:168], label='예측값', color='#ff7f0e', linestyle='--', linewidth=2)
plt.title('다변량 LSTM: 전력 사용량 예측')
plt.xlabel('시간')
plt.ylabel('전력 사용량(kW)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
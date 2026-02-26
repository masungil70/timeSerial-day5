import pandas as pd
import numpy as np
import koreanize_matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. 환경 설정
plt.rcParams['axes.unicode_minus'] = False 

# [단계 1] 데이터 로드
df = pd.read_csv('./data/power_usage_dataset.csv')
df['Date'] = pd.to_datetime(df['Date'])

# [단계 2] 특성 공학 (Feature Engineering)
# (1) 시간 주기성 변환
df['hour'] = df['Date'].dt.hour
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 23)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 23)

# (2) 주말 변수 추가 (토/일은 1, 평일은 0)
df['is_weekend'] = df['Date'].dt.weekday.map(lambda x: 1 if x >= 5 else 0)

# 사용할 특성 리스트 (총 5개)
features_list = ['Temperature', 'Usage', 'hour_sin', 'hour_cos', 'is_weekend']
data = df[features_list].values

# [단계 3] 데이터 전처리
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

def create_sequences(data, window_size=24):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size, :]) 
        y.append(data[i + window_size, 1]) # Target: Usage
    return np.array(X), np.array(y)

window_size = 24 # 과거 24시간 패턴 학습
X, y = create_sequences(scaled_data, window_size)

# 데이터 분할 (순서 유지)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# [단계 4] 최신 스타일 모델 설계 (Stacked LSTM)
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])), # (24, 5) 규격 명시
    LSTM(128, activation='tanh'),
    Dense(32, activation='tanh'),
    Dense(1)
])

# [단계 5] 컴파일 
model.compile(optimizer='adam', loss='mse')

# 조기 종료 설정
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=7, 
    restore_best_weights=True
)

# 모델 학습
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# [단계 6] 예측 및 역스케일링
predictions_scaled = model.predict(X_test)

def get_original_units(scaled_values, scaler, feature_count, target_idx=1):
    """5개 변수 구조를 유지하며 Target(전력량)만 역변환"""
    dummy = np.zeros((len(scaled_values), feature_count))
    dummy[:, target_idx] = scaled_values.flatten()
    return scaler.inverse_transform(dummy)[:, target_idx]

y_test_original = get_original_units(y_test, scaler, len(features_list))
predictions_original = get_original_units(predictions_scaled, scaler, len(features_list))

# [단계 7] 결과 시각화
plt.figure(figsize=(14, 6))
plt.plot(y_test_original[:168], label='실제값', color='#1f77b4', linewidth=2)
plt.plot(predictions_original[:168], label='예측값', color='#ff7f0e', linestyle='--', linewidth=2)
plt.title('다변량 LSTM (Weekend Feature): 스마트 기기 전력 사용량 예측')
plt.xlabel('시간 (1주일치)')
plt.ylabel('전력 사용량(kW)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
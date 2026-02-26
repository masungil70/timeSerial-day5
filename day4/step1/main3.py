import pandas as pd
import numpy as np
import koreanize_matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. 환경 설정 및 GPU 메모리 최적화
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

plt.rcParams['axes.unicode_minus'] = False 

# [단계 1] 데이터 로드
df = pd.read_csv('./data/power_usage_dataset.csv')
df['Date'] = pd.to_datetime(df['Date'])

# [단계 2] 특성 공학 (Cycle Encoding)
# 시간(0~23) 주기성
df['hour'] = df['Date'].dt.hour
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 23)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 23)

# 요일(0~6) 주기성: 금요일과 월요일의 인접성을 모델이 이해하게 함
df['weekday'] = df['Date'].dt.weekday
df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 6)
df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 6)

# 분석에 사용할 6개 필드
features_list = ['Temperature', 'Usage', 'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos']
data = df[features_list].values

# [단계 3] 데이터 전처리
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

def create_sequences(data, window_size=24):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size, :]) 
        y.append(data[i + window_size, 1]) # Target: Usage (Index 1)
    return np.array(X), np.array(y)

# 1주일(24)의 패턴을 보고 다음 1시간을 예측
window_size = 24 
X, y = create_sequences(scaled_data, window_size)

# 데이터 분할
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# [단계 4] 최신 스타일 모델 설계 (Input 레이어 명시)
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])), # (24, 6) 규격 명시
    LSTM(128, activation='tanh'),
    Dense(32, activation='tanh'),
    Dense(1)
])


# [단계 5] 컴파일 
model.compile(optimizer='adam', loss='mse')

# 조기 종료 설정
early_stop = EarlyStopping(
    monitor='val_loss',       # 감시 대상: 검증 데이터의 손실 값
    patience=10,               # 성능 개선이 없을 때 기다려줄 에포크 횟수
    restore_best_weights=True # 학습 종료 후 가장 성적이 좋았던 시점의 가중치로 복원
)

# 모델 학습
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64, # 3개월치 데이터이므로 배치 사이즈를 약간 키워 학습 안정성 확보
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# [단계 6] 예측 및 역스케일링
predictions_scaled = model.predict(X_test)

def get_original_units(scaled_values, scaler, feature_count, target_idx=1):
    """6개 피처 규격에 맞춰 역스케일링 수행"""
    dummy = np.zeros((len(scaled_values), feature_count))
    dummy[:, target_idx] = scaled_values.flatten()
    return scaler.inverse_transform(dummy)[:, target_idx]

y_test_original = get_original_units(y_test, scaler, len(features_list))
predictions_original = get_original_units(predictions_scaled, scaler, len(features_list))

# [단계 7] 결과 시각화 (최근 1주일치 결과 확인)
plt.figure(figsize=(14, 6))
plt.plot(y_test_original[:168], label='실제값', color='#1f77b4', linewidth=2)
plt.plot(predictions_original[:168], label='예측값', color='#ff7f0e', linestyle='--', linewidth=2)
plt.title('요일/시간 주기성이 반영된 다변량 LSTM 예측')
plt.xlabel('시간 (168시간 = 1주일)')
plt.ylabel('전력 사용량(kW)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
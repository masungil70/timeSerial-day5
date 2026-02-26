import pandas as pd
import numpy as np
import koreanize_matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# 1. GPU 메모리 동적 할당 (RTX 30 시리즈 등 최신 환경 필수)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU 설정 오류: {e}")

# 시각화 설정
plt.rcParams['axes.unicode_minus'] = False 

# [단계 1 & 2] 데이터 로드 및 특성 공학 (Cycle Encoding)
df = pd.read_csv('./data/power_usage_dataset_3month.csv')
df['Date'] = pd.to_datetime(df['Date'])

# 시간(0~23) 및 요일(0~6) 주기성 반영
df['hour'] = df['Date'].dt.hour
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 23)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 23)

df['weekday'] = df['Date'].dt.weekday
df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 6)
df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 6)

features_list = ['Temperature', 'Usage', 'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos']
data = df[features_list].values

# [단계 3] 데이터 전처리
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

def create_sequences(data, window_size=168):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size, :]) 
        y.append(data[i + window_size, 1]) # Target: Usage
    return np.array(X), np.array(y)

window_size = 168 
X, y = create_sequences(scaled_data, window_size)

# 데이터 분할 (8:2)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"✅ 학습 데이터 규격: {X_train.shape} (샘플 수, 타임스텝, 피처 수)")

# [단계 4] Stacked LSTM 모델 설계
model = Sequential([
    # (1) 입력 전용 레이어: 데이터의 '문지기' 역할
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    
    # (2) 첫 번째 LSTM: return_sequences=True로 설정하여 시퀀스 전체를 다음 층으로 전달
    # 과적합 방지를 위해 L2 규제와 Dropout 추가
    LSTM(64, activation='tanh', return_sequences=True, kernel_regularizer=l2(0.0001)),
    Dropout(0.2),
    
    # (3) 두 번째 LSTM: return_sequences=False로 설정하여 시퀀스를 요약된 벡터로 변환
    LSTM(32, activation='tanh', return_sequences=False, kernel_regularizer=l2(0.0001)),
    Dropout(0.1),
    
    # (4) 출력층
    Dense(1)
])

# [단계 5] 컴파일 및 학습
# optimizer를 객체로 호출하면 학습률(lr) 조정이 쉬워집니다.
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=7, 
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64, # 3개월치 데이터이므로 32보다 조금 키워 학습 속도 개선
    validation_split=0.1,
    callbacks=[early_stop],
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
plt.plot(y_test_original[:168], label='실제값 (Actual)', color='#1f77b4', linewidth=2)
plt.plot(predictions_original[:168], label='예측값 (Predicted)', color='#ff7f0e', linestyle='--', linewidth=2)
plt.title('최적화된 Stacked LSTM: 전력 사용량 예측 결과')
plt.xlabel('시간 (최근 1주일)')
plt.ylabel('전력 사용량(kW)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
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

# 1. GPU 메모리 설정 (RTX 30 시리즈 등 가속기 대응)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU 초기화 오류: {e}")

# 시각화 설정
plt.rcParams['axes.unicode_minus'] = False 

# [단계 1 & 2] 데이터 로드 및 특성 공학 (Cycle Encoding)
df = pd.read_csv('./data/power_usage_dataset_3month.csv')
df['Date'] = pd.to_datetime(df['Date'])

# 주기적 특성 인코딩 (Cyclic Encoding)
# 
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
        y.append(data[i + window_size, 1]) 
    return np.array(X), np.array(y)

window_size = 168 
X, y = create_sequences(scaled_data, window_size)

# 데이터 분할 (8:2)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# [단계 4] 스타일의 Stacked LSTM 설계
model = Sequential([
    # 명시적 입력층 정의 (window_size, feature_count)
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    
    # 첫 번째 LSTM: return_sequences=True 필수
    LSTM(64, activation='tanh', return_sequences=True, 
         kernel_regularizer=l2(0.001)),
    Dropout(0.2),
    
    # 두 번째 LSTM: 최종 특징 추출 (return_sequences=False)
    LSTM(32, activation='tanh', return_sequences=False, 
         kernel_regularizer=l2(0.001)),
    Dropout(0.2),
    
    Dense(1)
])

# [단계 5] 컴파일 및 학습
model.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=7, 
    restore_best_weights=True
)

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
    dummy = np.zeros((len(scaled_values), feature_count))
    dummy[:, target_idx] = scaled_values.flatten()
    return scaler.inverse_transform(dummy)[:, target_idx]

y_test_original = get_original_units(y_test, scaler, len(features_list))
predictions_original = get_original_units(predictions_scaled, scaler, len(features_list))

# [단계 7] 결과 시각화
plt.figure(figsize=(14, 6))
plt.plot(y_test_original[:168], label='실제값 (Actual)', color='#1f77b4', linewidth=2)
plt.plot(predictions_original[:168], label='예측값 (Predicted)', color='#ff7f0e', linestyle='--', linewidth=2)
plt.title('Stacked LSTM Model: 스마트 기기 전력 사용량 예측')
plt.xlabel('시간 (1주일 패턴)')
plt.ylabel('전력 사용량(kW)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# [단계 8] 학습 결과(Loss) 시각화
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('모델 학습 손실(Loss) 곡선')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.show()

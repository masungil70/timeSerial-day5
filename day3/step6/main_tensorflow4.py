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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision

# 4. GPU 하드웨어 가속 최적화 (XLA 컴파일)
# XLA(Accelerated Linear Algebra)는 모델의 연산 그래프를 분석하여 GPU에 
# 최적화된 형태로 '통합'해주는 컴파일러입니다.
# `model.compile` 시 옵션을 추가합니다.
# model.compile(optimizer='adam', loss='mse', jit_compile=True)

# 1. 하드웨어 가속 설정 (GPU & Mixed Precision)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 메모리 동적 할당
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # 혼합 정밀도 설정 (FP16 연산 가속)
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print(f"하드웨어 가속 활성화: {policy.name}")
    except RuntimeError as e:
        print(f"설정 오류: {e}")

# [단계 1] 데이터 로드
df = pd.read_csv('./data/power_usage_dataset_3month.csv')
df['Date'] = pd.to_datetime(df['Date'])

# [단계 2] 특성 공학 (Cycle Encoding)
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

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# [단계 4] 모델 설계 (현대적인 Stacked LSTM + Mixed Precision)
model = Sequential([
    # 명시적 입력 정의
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    
    # 가독성을 위한 수직 정렬 스타일
    LSTM(128, 
         activation='tanh', 
         return_sequences=True, 
         kernel_regularizer=l2(0.0001)),
    Dropout(0.2),
    
    LSTM(64, 
         activation='tanh', 
         return_sequences=False, 
         kernel_regularizer=l2(0.0001)),
    Dropout(0.1),
    
    # Mixed Precision 사용 시 출력층은 float32 필수
    Dense(1, dtype='float32')
])

# [단계 5] 컴파일 및 학습
# jit_compile=True: XLA 컴파일러 활성화 (연산 융합을 통한 추가 가속)
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', jit_compile=True)

early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=7, 
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=256,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# [단계 6] 학습 결과(Loss) 시각화 추가
# 모델이 과적합되는지, 안정적으로 학습되는지 확인하는 필수 과정입니다.
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('모델 학습 손실(Loss) 추이')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.show()

# [단계 7] 예측 및 역스케일링
predictions_scaled = model.predict(X_test)

def get_original_units(scaled_values, scaler, feature_count, target_idx=1):
    dummy = np.zeros((len(scaled_values), feature_count))
    dummy[:, target_idx] = scaled_values.flatten()
    return scaler.inverse_transform(dummy)[:, target_idx]

y_test_original = get_original_units(y_test, scaler, len(features_list))
predictions_original = get_original_units(predictions_scaled, scaler, len(features_list))

# [단계 8] 결과 시각화
plt.figure(figsize=(14, 6))
plt.plot(y_test_original[:168], label='실제값', color='#1f77b4', linewidth=2)
plt.plot(predictions_original[:168], label='예측값', color='#ff7f0e', linestyle='--', linewidth=2)
plt.title('Stacked LSTM (Mixed Precision + XLA): 전력 사용량 예측')
plt.xlabel('시간')
plt.ylabel('전력 사용량(kW)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
import pandas as pd
import numpy as np
import koreanize_matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision

# 1. 하드웨어 가속 설정
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # RTX 30 시리즈 이상 필수: 혼합 정밀도 활성화
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print(f"🚀 가속 정책 적용: {policy.name}")
    except RuntimeError as e:
        print(e)


# [단계 1 & 2] 데이터 로드 및 특성 공학 (Cycle Encoding)
df = pd.read_csv('./data/power_usage_dataset_3month.csv')
df['Date'] = pd.to_datetime(df['Date'])

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

# [단계 4] 모델 설계
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    
    LSTM(128, activation='tanh', return_sequences=True, kernel_regularizer=l2(0.0001)),
    Dropout(0.2),
    
    LSTM(64, activation='tanh', return_sequences=False, kernel_regularizer=l2(0.0001)),
    Dropout(0.1),
    
    # Mixed Precision 대응: 최종 출력층은 float32
    Dense(1, dtype='float32')
])

# [단계 5] 컴파일 및 학습 (XLA 컴파일 활성화)
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', jit_compile=True)

early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=256,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# [단계 6] 모델 및 스케일러 저장
# 폴더가 없으면 자동 생성하는 로직 추가
save_dir = './model'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 1) 모델 저장: 최신 Keras 방식인 .keras 확장자 권장 (또는 .h5)
model_path = os.path.join(save_dir, 'power_usage_lstm_model.h5')
model.save(model_path)

# 2) 스케일러 저장: joblib 사용
scaler_path = os.path.join(save_dir, 'power_usage_scaler.pkl')
joblib.dump(scaler, scaler_path)

print(f"\n✅ 저장 완료:")
print(f"   - 모델: {model_path} ({os.path.getsize(model_path)/(1024*1024):.2f} MB)")
print(f"   - 스케일러: {scaler_path}")
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

# ---------------------------------------------------------
# [데이터 전처리 : 단계 1] 데이터 로드 및 특성 공학
# ---------------------------------------------------------
df = pd.read_csv('./data/power_usage_dataset_3month.csv')
df['Date'] = pd.to_datetime(df['Date'])

# [데이터 전처리 : 단계 2] 이상치 탐지 및 제거
# 0 ~ 3.0 초과인 값을 찾아 NaN(결측치)으로 바꿉니다.
# 'Usage' 컬럼을 기준으로 처리하며, 데이터 특성에 따라 컬럼명을 확인해 주세요.
df.loc[(df['Usage'] < 0) | (df['Usage'] > 3.0), 'Usage'] = np.nan

# [데이터 전처리 : 단계 3] 선형 보간(Linear Interpolation) 수행
# NaN 앞뒤의 데이터를 연결하는 선을 그려 중간값을 채웁니다. 
# 시계열 데이터의 흐름을 깨지 않는 가장 표준적인 방법입니다.
df['Usage'] = df['Usage'].interpolate(method='linear')

# [데이터 전처리 : 단계 4] 잔여 결측치 처리
# 만약 데이터의 맨 첫 줄이나 맨 마지막 줄이 NaN이라면 보간되지 않을 수 있습니다.
# 이런 경우 근처의 값으로 채워(ffill, bfill) 완벽하게 결측치를 없앱니다.
df['Usage'] = df['Usage'].ffill().bfill()

# [데이터 전처리 : 단계 5] 특성 공학 (Feature Engineering)
# 시간 및 요일 주기성 반영 (Cyclic Encoding)
df['hour'] = df['Date'].dt.hour
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 23)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 23)

df['weekday'] = df['Date'].dt.weekday
df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 6)
df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 6)

# 분석 필드 (총 6개)
features_list = ['Temperature', 'Usage', 'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos']
data = df[features_list].values

# ---------------------------------------------------------
# [데이터 전처리 : 단계 6] 데이터 스케일링
# ---------------------------------------------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

def create_sequences(data, window_size=168):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size, :]) 
        y.append(data[i + window_size, 1]) # Target: Usage
    return np.array(X), np.array(y)

window_size = 168 # 1주일 패턴 학습
X, y = create_sequences(scaled_data, window_size)

# 데이터 분할 (8:2)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ---------------------------------------------------------
# [단계 2 : 모델 설계] 최신 Keras 스타일 모델 설계 (Input 레이어 명시)
# ---------------------------------------------------------
model = Sequential([
    # 명시적 입력 정의: (타임스텝, 피처수)
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    
    # 첫 번째 LSTM 계층: L2 규제 및 드롭아웃
    LSTM(128
         , activation='tanh'
         , return_sequences=True
         , kernel_regularizer=l2(0.0001)),
    Dropout(0.2),
    
    # 두 번째 LSTM 계층
    LSTM(64
         , activation='tanh'
         , return_sequences=False
         , kernel_regularizer=l2(0.0001)),
    Dropout(0.1),
    
    # 출력 계층: 혼합 정밀도 대응을 위해 float32 명시
    Dense(1, dtype='float32')
])

# ---------------------------------------------------------
# [단계 3] 컴파일 및 학습 (XLA 적용)
# ---------------------------------------------------------
optimizer = Adam(learning_rate=0.001)

# jit_compile=True: GPU 하드웨어 가속 최적화 (XLA 컴파일러)
model.compile(optimizer=optimizer, loss='mse', jit_compile=True)

early_stop = EarlyStopping(
    monitor='val_loss',         # 감시 대상: 검증 데이터의 손실 값
    patience=7,                 # 성능 개선이 없을 때 기다려줄 에포크 횟수
    restore_best_weights=True   # 학습 종료 후 가장 성적이 좋았던 시점의 가중치로 복원
)

print("\n🚀 모델 학습을 시작합니다...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=256, # 대량 데이터 및 GPU 가속을 위한 큰 배치 사이즈
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# ---------------------------------------------------------
# [단계 4] 예측 및 역스케일링
# ---------------------------------------------------------
predictions_scaled = model.predict(X_test)

def get_original_units(scaled_values, scaler, feature_count, target_idx=1):
    dummy = np.zeros((len(scaled_values), feature_count))
    dummy[:, target_idx] = scaled_values.flatten()
    return scaler.inverse_transform(dummy)[:, target_idx]

y_test_original = get_original_units(y_test, scaler, len(features_list))
predictions_original = get_original_units(predictions_scaled, scaler, len(features_list))

# ---------------------------------------------------------
# [단계 5] 시각화 및 저장
# ---------------------------------------------------------
# 1. 예측 결과 시각화
plt.figure(figsize=(14, 6))
plt.plot(y_test_original[:168], label='실제값', color='#1f77b4', linewidth=2)
plt.plot(predictions_original[:168], label='예측값', color='#ff7f0e', linestyle='--', linewidth=2)
plt.title('최적화된 Stacked LSTM: 전력 사용량 예측 결과 (1주일)')
plt.xlabel('시간')
plt.ylabel('전력 사용량(kW)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 2. 모델 및 스케일러 저장
save_dir = './model'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model_path = os.path.join(save_dir, 'power_usage_lstm_model.keras')
scaler_path = os.path.join(save_dir, 'power_usage_scaler.pkl')

model.save(model_path)
joblib.dump(scaler, scaler_path)

print(f"\n✅ 완료: 모델({model_path}) 및 스케일러({scaler_path}) 저장 성공!")
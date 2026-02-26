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

# 혼합 정밀도(Mixed Precision) 사용 : 16비트와 32비트 연산 혼합
# GPU는 Tensor Core라는 딥러닝 전용 가속기가 있습니다.
# 기본 32비트 연산을 16비트로 낮춰 연산하면 속도는 2배 빨라지고
# 메모리 사용량은 절반으로 줄어듭니다.

# 1. GPU 메모리 설정 및 혼합 정밀도 활성화
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 메모리 동적 할당
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # [최신 기법] 혼합 정밀도 정책 설정 (RTX 30 시리즈 전용)
        # float16으로 연산하되, 수치 불안정 방지를 위해 가중치는 float32로 관리함
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print("Mixed precision policy set to: mixed_float16")
    except RuntimeError as e:
        print(f"GPU 초기화 오류: {e}")

# [단계 1] 데이터 처리
df = pd.read_csv('./data/power_usage_dataset_3month.csv')
df['Date'] = pd.to_datetime(df['Date'])

# [단계 2] 특성 공학 (Sine/Cosine Encoding)
# 시간의 주기성을 수학적으로 모델에게 전달 (23시와 0시의 연결성)
df['hour'] = df['Date'].dt.hour
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 23)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 23)

df['weekday'] = df['Date'].dt.weekday
df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 6)
df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 6)

features_list = ['Temperature', 'Usage', 'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos']
data = df[features_list].values

# [단계 3] 스케일링 및 시퀀스 생성
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

# [단계 4] 최신 스타일 모델 설계 (Input 레이어 도입)
model = Sequential([
    # 명시적 입력층 정의
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    
    # 첫 번째 LSTM: return_sequences=True 필수 (다음 층도 LSTM이므로)
    LSTM(
        units=128, 
        activation='tanh', 
        return_sequences=True,
        kernel_regularizer=l2(0.0001)
    ),
    Dropout(0.2),
    
    # 두 번째 LSTM: 최종 벡터 추출
    LSTM(
        units=64, 
        activation='tanh', 
        return_sequences=False,
        kernel_regularizer=l2(0.0001)
    ),
    Dropout(0.1),
    
    # [주의] 출력층: Mixed Precision 사용 시 dtype='float32'로 강제해야 함
    # 최종 예측값의 오차 계산 시 수치 정밀도를 확보하기 위함
    Dense(1, dtype='float32')
])

# [단계 5] 컴파일 및 대규모 배치 학습
# 배치 사이즈(256)를 키웠으므로 학습률을 기본(0.001)보다 약간 높이거나 유지하는 것이 좋습니다.
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse')

early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=7, 
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=100,      # 학습 횟수 확대
    batch_size=256,  # 대량 병렬 처리
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

# [단계 7] 시각화
plt.figure(figsize=(14, 6))
plt.plot(y_test_original[:168], label='실제 사용량', color='#1f77b4', linewidth=2)
plt.plot(predictions_original[:168], label='LSTM 예측값', color='#ff7f0e', linestyle='--', linewidth=2)
plt.title('Stacked LSTM: 전력 사용량 예측 (Mixed Precision 활성)')
plt.xlabel('시간 (Time)')
plt.ylabel('전력 사용량 (kW)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
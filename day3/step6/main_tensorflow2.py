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

#batch_size를 크게 설정하기 하여 성능 향상하기
#수정: `32` → **`256`** 또는 **`512`**
#        batch_size의 값을 무조건 크게 설정한다고 해서 항상 좋은 것은 아닙니다.
#       32, 64, 128, 256 까지는 결과가 비슷하지만 
#       512 이상부터는 메모리 부족 현상, 결과 값이 다르게 나올 수 있습니다
#       output 폴더에 수치 변경에 따른 결과 파일들을 참고하세요.
#           
#효과: 한 번에 처리하는 데이터 양이 늘어나 GPU의 수천 개 CUDA 코어를 꽉 채워 쓸 수 있습니다. 에포크당 소요 시간이 대폭 줄어듭니다.
#주의: 배치 사이즈를 키우면 한 에포크당 가중치 업데이트 횟수(Iteration)가 줄어드므로, 학습률(`lr`)을 조금 높여주는 것이 좋습니다.
#      # [단계 5] 모델 컴파일 및 학습
#      optimizer='adam' 학습률 기본값 0.001을 의미하는 것입니다
#      model.compile(optimizer=optimizer, loss='mse')
#
#      학습률을 0.002로 높여서 설정 (기본값보다 2배 빠른 보폭)
#      optimizer = Adam(learning_rate=0.002)
#      model.compile(optimizer=optimizer, loss='mse')
#


# 1. GPU 메모리 관리 (RTX 30 시리즈 등 최신 그래픽카드 최적화)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # GPU 메모리를 한꺼번에 점유하지 않고 필요한 만큼 동적으로 할당
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU 설정 오류: {e}")

# [단계 1] 데이터 로드
df = pd.read_csv('./data/power_usage_dataset_3month.csv')
df['Date'] = pd.to_datetime(df['Date'])

# [단계 2] 특성 공학 (Feature Engineering)
# 시간 데이터의 연속성을 Sine/Cosine으로 변환 (23시와 0시가 가깝다는 수학적 정보를 제공)
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
        y.append(data[i + window_size, 1]) # Target: Usage (인덱스 1번)
    return np.array(X), np.array(y)

window_size = 168 # 1주일치 패턴 학습
X, y = create_sequences(scaled_data, window_size)

# 데이터셋 분할 (8:2)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# [단계 4] 모델 설계 (최신 권장 스타일: Input 레이어 명시)
model = Sequential([
    # (1) 입력 전용 층: 데이터 규격을 명확히 선언
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    
    # (2) 첫 번째 LSTM: return_sequences=True로 설정하여 다음 LSTM에 시퀀스 전달
    LSTM(
        units=128, 
        activation='tanh', 
        return_sequences=True,
        kernel_regularizer=l2(0.0001) # 가중치 크기를 제한하여 과적합 방지
    ),
    Dropout(0.2), # 학습 시 뉴런 20%를 무작위로 꺼서 일반화 성능 향상
    
    # (3) 두 번째 LSTM: 최종 특징 추출 (return_sequences=False)
    LSTM(
        units=64, 
        activation='tanh', 
        return_sequences=False,
        kernel_regularizer=l2(0.0001)
    ),
    Dropout(0.1),
    
    # (4) 출력층
    Dense(1)
])

# [단계 5] 컴파일 및 학습 설정
# 하이퍼파라미터 최적화: 보폭(LR)을 높이고, 한 번에 보는 양(Batch)을 늘림
optimizer = Adam(learning_rate=0.002) 
model.compile(optimizer=optimizer, loss='mse')

# 검증 손실(val_loss)이 7회 이상 개선되지 않으면 학습 중단
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=7, 
    restore_best_weights=True # 가장 좋았던 상태의 가중치로 복구
)

# 모델 학습
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=256,   # GPU 병렬 연산을 극대화하기 위해 배치 크기 확대
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# [단계 6] 예측 및 역스케일링
predictions_scaled = model.predict(X_test)

def get_original_units(scaled_values, scaler, feature_count, target_idx=1):
    """MinMax 스케일링된 데이터를 실제 전력 사용량 단위로 복원"""
    dummy = np.zeros((len(scaled_values), feature_count))
    dummy[:, target_idx] = scaled_values.flatten()
    return scaler.inverse_transform(dummy)[:, target_idx]

y_test_original = get_original_units(y_test, scaler, len(features_list))
predictions_original = get_original_units(predictions_scaled, scaler, len(features_list))

# [단계 7] 시각화
plt.figure(figsize=(14, 6))
plt.plot(y_test_original[:168], label='실제값 (Actual)', color='#1f77b4', linewidth=2)
plt.plot(predictions_original[:168], label='예측값 (Predicted)', color='#ff7f0e', linestyle='--', linewidth=2)
plt.title('Stacked LSTM: 전력 사용량 예측 결과 (1주일 패턴)')
plt.xlabel('시간 (Time)')
plt.ylabel('전력 사용량 (kW)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
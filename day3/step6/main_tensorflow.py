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

# 1. GPU 메모리 동적 할당 (RTX 30 시리즈 이상 권장 설정)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 필요한 만큼의 메모리만 할당하도록 설정
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# [단계 1 & 2] 데이터 로드 및 시계열 특성 추출 (Cycle Encoding)
df = pd.read_csv('./data/power_usage_dataset_3month.csv')
df['Date'] = pd.to_datetime(df['Date'])

# 시간/요일의 연속성을 위해 Sine/Cosine 변환 적용 (23시와 0시가 가깝다는 것을 모델에게 알려줌)
df['hour'] = df['Date'].dt.hour
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 23)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 23)

df['weekday'] = df['Date'].dt.weekday
df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 6)
df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 6)

# 학습에 사용할 특성 선택
features_list = ['Temperature', 'Usage', 'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos']
data = df[features_list].values

# [단계 3] 데이터 스케일링 및 시퀀스 생성
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

def create_sequences(data, window_size=168):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size, :]) 
        y.append(data[i + window_size, 1]) # Target: Usage (Index 1)
    return np.array(X), np.array(y)

window_size = 168 
X, y = create_sequences(scaled_data, window_size)

# 훈련/테스트 데이터 분할 (8:2)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# [단계 4] 최신 스타일의 모델 설계
# 1) Input 레이어를 분리하여 규격을 명시함
# 2) 매개변수를 수직으로 정렬하여 가독성 확보
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])), # (168, 6) 형태의 입력 정의
    
    # 첫 번째 LSTM: 고차원 특징 추출 (return_sequences=True 필수)
    LSTM(
        units=128, 
        activation='tanh', 
        return_sequences=True,
        kernel_regularizer=l2(0.0001)
    ),
    Dropout(0.2),
    
    # 두 번째 LSTM: 시퀀스 정보를 압축하여 최종 특징 추출 (return_sequences=False)
    LSTM(
        units=64, 
        activation='tanh', 
        return_sequences=False,
        kernel_regularizer=l2(0.0001)
    ),
    Dropout(0.1),
    
    # 출력층: 전력 사용량 1개 값 예측
    Dense(1)
])

# [단계 5] 컴파일 및 학습 설정
# optimizer 객체를 직접 호출하면 learning_rate 조절이 용이함 (권장)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
    loss='mse'
)

# 조기 종료 설정: val_loss가 7번 동안 개선되지 않으면 멈춤
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=7, 
    restore_best_weights=True
)

# 모델 학습 실행
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1, # 훈련 데이터의 10%를 검증에 사용
    callbacks=[early_stop],
    verbose=1
)

# [단계 6] 예측 및 역스케일링 (Inverse Transform)
predictions_scaled = model.predict(X_test)

def get_original_units(scaled_values, scaler, feature_count, target_idx=1):
    """스케일링된 데이터를 원래 단위로 복구하는 함수"""
    dummy = np.zeros((len(scaled_values), feature_count))
    dummy[:, target_idx] = scaled_values.flatten()
    return scaler.inverse_transform(dummy)[:, target_idx]

y_test_original = get_original_units(y_test, scaler, len(features_list))
predictions_original = get_original_units(predictions_scaled, scaler, len(features_list))

# [단계 7] 시각화
plt.figure(figsize=(14, 6))
plt.plot(y_test_original[:168], label='Actual Usage', color='#1f77b4', linewidth=2)
plt.plot(predictions_original[:168], label='Predicted Usage', color='#ff7f0e', linestyle='--', linewidth=2)
plt.title('Stacked LSTM Model: 스마트 기기 전력 사용량 예측')
plt.xlabel('시간')
plt.ylabel('전력 사용량(kW)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
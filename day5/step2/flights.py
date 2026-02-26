import pandas as pd
import numpy as np
import koreanize_matplotlib
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima

# [1] 데이터 로드
file_path = './data/flights.csv'
# parse_dates=['Month']: 'Month' 컬럼을 문자열이 아닌 시계열 날짜 객체로 읽어옴
df = pd.read_csv(file_path, index_col='Month', parse_dates=True)

# [2] 시계열 데이터 분해 (Seasonal Decomposition)
# - 시계열 데이터를 구성하는 3요소(추세, 계절성, 잔차)를 시각적으로 분리하여 확인
# - model='additive': 가법 모델 (Trend + Seasonal + Resid)
# - period=월별 데이터이므로 1년 주기는 12입니다.
print("데이터 분해를 시작합니다 (Trend, Seasonal, Resid 확인)...")
result = seasonal_decompose(df['Passengers'], model='additive', period=12)

# 결과 그래프 출력: 추세(상승/하강), 계절성(패턴), 잔차(불규칙 변동)를 한눈에 볼 수 있음
result.plot()
plt.show()

# [3] 최적의 ARIMA 모델 탐색 (auto_arima)
# - ARIMA(p, d, q) 모델의 최적 파라미터를 자동으로 찾아주는 알고리즘
# - m=7: 계절성 주기 (여기서는 7일 단위의 주간 패턴)
# - seasonal=True: SARIMA(Seasonal ARIMA)를 적용하여 계절적 요인 반영
# - stepwise=True: 연산량을 줄이기 위해 최적의 파라미터를 단계별로 검색
print("최적의 ARIMA 파라미터를 찾는 중입니다 (자동 탐색 단계)...")
stepwise_model = auto_arima(df['Passengers'], 
                            m=12,                
                            seasonal=True,      
                            stepwise=True,      
                            trace=False         
                            )

# 탐색 완료 후 최적의 모델 요약 정보 출력
print(stepwise_model.summary())

# [4] 학습 및 미래 예측 테스트
# - 마지막 365일을 테스트용 데이터로 사용하여 모델의 정확도 검증
train = df['Passengers'][:-12]
test = df['Passengers'][-12:]

# 훈련 데이터로 모델 학습
stepwise_model.fit(train)

# 학습된 모델로 테스트 데이터 기간(12개월)만큼 미래 예측
future_forecast = stepwise_model.predict(n_periods=12)

# [5] 예측 결과 시각화
# - 실제 데이터(Actual)와 모델의 예측값(Prediction)을 비교
plt.figure(figsize=(12, 6))
plt.plot(test.index, test.values, label='Actual (실제)')
plt.plot(test.index, future_forecast, label='Auto-ARIMA (추론)', color='red')
plt.title('비행기 승객 수 추론 (Auto-ARIMA)')
plt.legend()
plt.grid(True)
plt.show()
# ==============================
# 0. 필수 라이브러리 불러오기
# ==============================

# 기본 수치 계산 및 데이터 조작
import numpy as np

# 그래프 및 시각화 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns

# MOABB (EEG 데이터셋 라이브러리)에서 MAMEM3 데이터셋 및 SSVEP 파라다임 불러오기
from moabb.datasets import MAMEM3
from moabb.paradigms import SSVEP

# 데이터 전처리 및 평가 관련 라이브러리
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight  # 데이터 불균형 해소용

# TensorFlow / Keras - 모델 구성 및 최적화
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, SeparableConv2D, Dense, Flatten, Dropout, BatchNormalization, Activation, AveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# EEG 신호 필터링용 라이브러리
from scipy.signal import butter, lfilter

# 한글 폰트 설정 및 그래프 스타일
import matplotlib as mpl
mpl.rc('font', family='AppleGothic')  # Mac 사용시 한글 폰트 설정
mpl.rcParams['axes.unicode_minus'] = False  # 음수 표시 설정

# ==============================
# 1. MAMEM3 데이터셋 불러오기
# ==============================

# MAMEM3 SSVEP 데이터셋 불러오기
dataset = MAMEM3()
paradigm = SSVEP(n_classes=4, channels=None)  # SSVEP 파라다임, 4개의 클래스를 사용

# EEG 데이터 로드 (특정 피험자 선택)
X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[1])
print("데이터셋 크기:", X.shape)  # (샘플 수, 채널 수, 시간 샘플 수)
print("레이블 분포:", np.unique(y, return_counts=True))  # 각 클래스별 데이터 개수

# ==============================
# 2. 데이터 전처리 (개선)
# ==============================

# --- 1) 밴드패스 필터링 (6~12Hz) ---
# EEG 신호의 특정 주파수 범위(SSVEP)를 강조하기 위한 필터링

def butter_bandpass(lowcut, highcut, fs, order=4):
    """버터워스 밴드패스 필터 생성 함수"""
    nyq = 0.5 * fs  # 나이퀴스트 주파수
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=6, highcut=12, fs=128, order=4):
    """EEG 데이터에 밴드패스 필터 적용"""
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return lfilter(b, a, data)

# EEG 신호 필터링 적용 (6~12Hz)
X_filtered = np.array([bandpass_filter(epoch, lowcut=6, highcut=12, fs=128) for epoch in X])

# --- 2) 신호 정규화 ---
# 각 채널별로 평균 0, 표준편차 1로 정규화 (표준화)
X_filtered = (X_filtered - np.mean(X_filtered, axis=-1, keepdims=True)) / np.std(X_filtered, axis=-1, keepdims=True)

# --- 3) 레이블 인코딩 ---
# 문자열 형태의 레이블을 정수로 변환 (e.g., '10.00' → 0)
le = LabelEncoder()
y_int = le.fit_transform(y)
print("인코딩된 레이블:", y_int)
print("클래스 매핑:", dict(zip(le.classes_, le.transform(le.classes_))))  # 매핑 정보 출력

# --- 4) 원-핫 인코딩 ---
# 다중 클래스 분류를 위해 원-핫 인코딩 적용
y_cat = to_categorical(y_int)

# --- 5) 훈련/테스트 데이터 분할 ---
# 훈련 데이터 80%, 테스트 데이터 20%로 분할
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_cat, test_size=0.2, random_state=42)

# --- 6) 채널 차원 추가 ---
# Conv2D 레이어에 입력하기 위해 채널 차원을 추가 (4D 텐서로 변환)
X_train = X_train[..., np.newaxis]  # (샘플, 채널, 시간, 1)
X_test = X_test[..., np.newaxis]

# --- 7) 클래스 가중치 계산 ---
# 데이터 불균형 문제 해결을 위한 클래스 가중치 계산
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_int), y=y_int)
class_weights_dict = dict(enumerate(class_weights))
print("클래스 가중치:", class_weights_dict)

# ==============================
# 3. EEGNet 모델 정의 (개선)
# ==============================

def EEGNet(nb_classes, Chans=14, Samples=384, dropoutRate=0.5, l2_reg=0.05):
    """
    EEGNet 아키텍처 정의
    - nb_classes: 분류할 클래스 수
    - Chans: EEG 채널 수
    - Samples: 시간 샘플 수
    """
    input1 = Input(shape=(Chans, Samples, 1))  # 입력 레이어

    # --- 시간적 컨볼루션 ---
    block1 = Conv2D(16, (1, 64), padding='same', use_bias=False,
                    kernel_regularizer=l2(l2_reg))(input1)
    block1 = BatchNormalization()(block1)

    # --- 공간적 컨볼루션 ---
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=2,
                             depthwise_regularizer=l2(l2_reg))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = Dropout(dropoutRate)(block1)

    # --- Separable Convolution ---
    block2 = SeparableConv2D(32, (1, 16), use_bias=False, padding='same',
                             depthwise_regularizer=l2(l2_reg),
                             pointwise_regularizer=l2(l2_reg))(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = Dropout(dropoutRate)(block2)

    # --- 출력 레이어 ---
    flatten = Flatten()(block2)
    dense = Dense(nb_classes, activation='softmax')(flatten)

    return Model(inputs=input1, outputs=dense)

# ==============================
# 4. EEGNet 모델 학습 (개선)
# ==============================

# 하이퍼파라미터 설정
nb_classes = y_cat.shape[1]  # 클래스 수
Chans = X.shape[1]  # EEG 채널 수
Samples = X.shape[2]  # 시간 샘플 수

# EEGNet 모델 생성
model = EEGNet(nb_classes, Chans, Samples)

# --- 모델 컴파일 ---
# Adam 옵티마이저 사용, 학습률 0.0001
optimizer = Adam(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# --- 조기 종료 콜백 ---
# 검증 손실이 개선되지 않으면 학습 중단 (patience=15)
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# --- 모델 학습 ---
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=8,
    validation_data=(X_test, y_test),
    class_weight=class_weights_dict,  # 불균형 데이터 가중치 적용
    callbacks=[early_stop],
    verbose=2
)

# ==============================
# 5. 성능 평가 및 시각화 (개선)
# ==============================

# --- 예측 ---
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # 예측 클래스
y_true = np.argmax(y_test, axis=1)  # 실제 클래스

# --- 평가 지표 계산 ---
acc = accuracy_score(y_true, y_pred_classes)  # 정확도
f1 = f1_score(y_true, y_pred_classes, average='weighted')  # F1 스코어
print(f"테스트 정확도: {acc:.2f}")
print(f"F1 Score: {f1:.2f}")

# ==============================
# 6. 학습 결과 시각화
# ==============================

# --- 학습 곡선 (정확도 & 손실) ---
plt.figure(figsize=(12,5))

# 정확도 그래프
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='훈련 정확도')
plt.plot(history.history['val_accuracy'], label='검증 정확도')
plt.xlabel('에폭')
plt.ylabel('정확도')
plt.title('정확도 변화')
plt.legend()

# 손실 그래프
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='훈련 손실')
plt.plot(history.history['val_loss'], label='검증 손실')
plt.xlabel('에폭')
plt.ylabel('손실')
plt.title('손실 변화')
plt.legend()

plt.tight_layout()
plt.show()

# ==============================
# 7. 혼동 행렬 시각화
# ==============================

# --- 혼동 행렬 ---
cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('예측 레이블')
plt.ylabel('실제 레이블')
plt.title(f'혼동 행렬 (정확도: {acc:.2%})')
plt.show()

# ==============================
# 8. 분류 보고서 출력
# ==============================

# --- 분류 성능 지표 ---
print("\n분류 보고서:\n", classification_report(y_true, y_pred_classes, target_names=le.classes_))
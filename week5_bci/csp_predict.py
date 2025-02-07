# 필요한 라이브러리 임포트
import numpy as np  # numpy 추가
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, train_test_split
import moabb
from moabb.datasets import Cho2017
from moabb.paradigms import LeftRightImagery

# MOABB 로그 레벨 설정
moabb.set_log_level("info")

# Cho2017 데이터셋 로드 및 Paradigm 설정
dataset = Cho2017()  # Cho2017 데이터셋 사용
paradigm = LeftRightImagery(resample=128)  # 샘플링 속도를 128Hz로 설정

# 데이터 로드 (특정 Subject 선택)
X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[1])  # Subject 1의 데이터를 사용
print("데이터 형태:", X.shape)
print("레이블 형태:", y.shape)
print("메타데이터 샘플:", metadata.head())

# 학습 및 테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CSP + LDA 파이프라인 구성
csp = CSP(n_components=8, reg=None, log=True)  # CSP 구성 (8개의 컴포넌트 사용)
lda = LDA()  # LDA 구성
pipeline = make_pipeline(csp, lda)  # CSP와 LDA를 파이프라인으로 결합

# 교차 검증 수행 (5-Fold)
scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
print("교차 검증 정확도 (5-Fold):", scores)
print("평균 교차 검증 정확도:", scores.mean())

# 모델 학습 및 테스트
pipeline.fit(X_train, y_train)  # 학습
test_score = pipeline.score(X_test, y_test)  # 테스트 데이터 평가
print("테스트 정확도:", test_score)

# --- 추가: 시각화를 위한 데이터 변환 ---
# CSP 변환 후 데이터 추출
X_csp = csp.fit_transform(X_train, y_train)  # CSP 변환 수행
X_test_csp = csp.transform(X_test)          # 테스트 데이터 변환

# LDA를 통한 분류
lda.fit(X_csp, y_train)             # LDA 학습
X_lda = lda.transform(X_csp)        # LDA 변환
X_test_lda = lda.transform(X_test_csp)

plt.figure(figsize=(12, 6))
colors = ['red', 'blue']
markers = ['o', 'x']
labels = np.unique(y_train)

# 학습 데이터 시각화
for i, label in enumerate(labels):
    indices = np.where(y_train == label)
    plt.scatter(X_lda[indices], [i] * len(indices[0]),
                color=colors[i], label=f'Class {label} (Train)', alpha=0.7, marker=markers[0])

# 테스트 데이터 시각화
for i, label in enumerate(labels):
    indices = np.where(y_test == label)
    plt.scatter(X_test_lda[indices], [i + 0.2] * len(indices[0]),
                color=colors[i], label=f'Class {label} (Test)', alpha=0.7, marker=markers[1])

# 그래프 설정
plt.title('CSP + LDA Classification Visualization (1D)', fontsize=16)
plt.xlabel('LDA Component 1', fontsize=12)
plt.ylabel('Class', fontsize=12)
plt.yticks(range(len(labels)), [f'Class {label}' for label in labels])
plt.legend(fontsize=10)
plt.grid(True)
plt.show()
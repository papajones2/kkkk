import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

CHANNELS = [
    '410nm', '435nm', '460nm', '485nm', '510nm', '535nm',
    '560nm', '585nm', '610nm', '645nm', '680nm', '705nm',
    '730nm', '760nm', '810nm', '860nm', '900nm', '940nm'
]

df = pd.read_csv("plastic_data_average2.csv")
labels = df['label'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

# ================================================================
# 1. 종류별 평균 스펙트럼 (raw ratio)
# raw ratio가 종류마다 다른 모양이면 → 데이터 자체는 괜찮은 것
# 모든 종류가 비슷한 모양이면 → 센서로 구별이 어려운 것
# ================================================================
plt.figure(figsize=(12, 5))
for label, color in zip(labels, colors):
    subset = df[df['label'] == label][CHANNELS]
    mean = subset.mean()
    std = subset.std()
    plt.plot(CHANNELS, mean, label=label, color=color, marker='o', markersize=4)
    plt.fill_between(CHANNELS,
                     mean - std,
                     mean + std,
                     alpha=0.1, color=color)  # 표준편차 범위

plt.xticks(rotation=45)
plt.xlabel('파장')
plt.ylabel('ratio (raw/baseline)')
plt.title('1. 종류별 평균 스펙트럼\n(선이 잘 분리될수록 구별 쉬움, 음영=표준편차)')
plt.legend()
plt.tight_layout()
plt.savefig('viz_1_spectrum.png')
plt.show()

# ================================================================
# 2. 채널별 boxplot
# 특정 파장에서 종류별 분포가 겹치지 않으면 → 그 파장이 핵심
# ================================================================
fig, axes = plt.subplots(3, 6, figsize=(18, 10))
axes = axes.flatten()

for idx, ch in enumerate(CHANNELS):
    data_per_label = [df[df['label'] == label][ch].values for label in labels]
    axes[idx].boxplot(data_per_label, labels=labels)
    axes[idx].set_title(ch)
    axes[idx].tick_params(axis='x', rotation=45)

plt.suptitle('2. 채널별 종류 분포\n(박스가 겹치지 않을수록 구별 쉬운 파장)', y=1.02)
plt.tight_layout()
plt.savefig('viz_2_boxplot.png')
plt.show()

# ================================================================
# 3. PCA (전처리 후)
# 덩어리가 잘 분리되면 → 분류기가 잘 작동할 것
# 섞여있으면 → 데이터 더 필요하거나 구별 자체가 어려운 것
# ================================================================
X = df[CHANNELS].values
y = df['label'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
for label, color in zip(labels, colors):
    mask = y == label
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                label=label, color=color, alpha=0.6)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.title('3. PCA (전처리 후)\n(덩어리가 잘 분리될수록 분류 정확도 높음)')
plt.legend()
plt.tight_layout()
plt.savefig('viz_3_pca.png')
plt.show()

# ================================================================
# 4. 상관관계 히트맵
# 어떤 채널끼리 비슷하게 움직이는지 확인
# ================================================================
plt.figure(figsize=(10, 8))
corr = df[CHANNELS].corr()
im = plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im)
plt.xticks(range(len(CHANNELS)), CHANNELS, rotation=45)
plt.yticks(range(len(CHANNELS)), CHANNELS)
plt.title('4. 채널간 상관관계\n(비슷한 채널은 하나로 줄여도 됨)')
plt.tight_layout()
plt.savefig('viz_4_correlation.png')
plt.show()
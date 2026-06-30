import pandas as pd
import numpy as np

# averaged 데이터 또는 원본 사용
try:
    df = pd.read_csv('plastic_data_averaged.csv')
    print("📊 averaged 데이터 사용\n")
except:
    df = pd.read_csv('plastic_data.csv')
    print("📊 원본 데이터 사용 (averaged 없음)\n")

print(f"총 데이터: {len(df)}행")
print(f"물질: {df['label'].value_counts().to_dict()}\n")

# 채널 컬럼들
channel_cols = [col for col in df.columns if col != 'label']

# 라벨별 오차 분석
for label in df['label'].unique():
    subset = df[df['label'] == label]
    print(f"\n{'='*60}")
    print(f"물질: {label} ({len(subset)}개 샘플)")
    print(f"{'='*60}")

    if len(subset) > 1:
        # 채널별 표준편차
        stds = []
        for col in channel_cols:
            std = np.std(subset[col].values, ddof=1)
            stds.append(std)

        stds = np.array(stds)

        print(f"\n📈 표준편차 통계:")
        print(f"   최소: {stds.min():.6f}")
        print(f"   최대: {stds.max():.6f}")
        print(f"   평균: {stds.mean():.6f}")
        print(f"   중앙값: {np.median(stds):.6f}")

        # 데이터 범위
        all_values = subset[channel_cols].values.flatten()
        print(f"\n📊 데이터 값 범위:")
        print(f"   최소: {all_values.min():.6f}")
        print(f"   최대: {all_values.max():.6f}")
        print(f"   평균: {all_values.mean():.6f}")

        # 추천 노이즈 설정
        rec_noise = stds.mean()
        print(f"\n✅ 추천 노이즈 설정 (평균 표준편차):")
        print(f"   ±{rec_noise:.6f} (±1σ)")
        print(f"   ±{rec_noise*1.5:.6f} (±1.5σ, 더 강함)")
        print(f"   ±{rec_noise*2:.6f} (±2σ, 매우 강함)")

    else:
        print(f"⚠️  샘플이 1개뿐이라 표준편차 계산 불가")

print(f"\n{'='*60}")
print("💡 권장:")
print(f"   - 노이즈: 평균 표준편차 × 1.0 ~ 1.5")
print(f"   - 증강: 원본 데이터 × 2~3배")
print(f"   - 예: 25개 샘플 → 50~75개")

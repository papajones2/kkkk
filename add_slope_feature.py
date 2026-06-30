import pandas as pd
import numpy as np

df = pd.read_csv('plastic_data.csv')

# 기울기 계산: 900nm, 940nm (인덱스 16, 17) / 410nm, 435nm (인덱스 0, 1)
ir_mean = (df.iloc[:, 16] + df.iloc[:, 17]) / 2.0
vis_mean = (df.iloc[:, 0] + df.iloc[:, 1]) / 2.0
slope = ir_mean / vis_mean

# 새 컬럼 추가
df.insert(18, 'slope_IR_VIS', slope)

# 저장
df.to_csv('plastic_data.csv', index=False)
print(f"✅ slope_IR_VIS 컬럼 추가 완료 ({len(df)}행)")
print(f"\n스펙트럼 기울기 통계:")
print(f"전체 - 평균: {slope.mean():.3f}, 표준편차: {slope.std():.3f}")
print(f"\nPP:  {slope[df['label']=='PP'].describe()}")
if 'PET' in df['label'].values:
    print(f"PET: {slope[df['label']=='PET'].describe()}")

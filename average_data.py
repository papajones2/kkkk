import pandas as pd
import numpy as np

df = pd.read_csv('plastic_data.csv')
print(f"원본 데이터: {len(df)}행")
print(f"물질 종류: {df['label'].value_counts().to_dict()}\n")

# 사용자 입력
measure_count = int(input("한 샘플당 몇 회씩 측정했나요? (예: 50, 25): ").strip())

# 채널 컬럼들 (label 제외)
channel_cols = [col for col in df.columns if col != 'label']

# measure_count개씩 묶어서 평균
new_rows = []
for i in range(0, len(df), measure_count):
    group_data = df.iloc[i:i+measure_count]

    # 모두 같은 label인지 확인
    if len(group_data['label'].unique()) == 1:
        # 채널별 평균
        avg_values = group_data[channel_cols].mean().values
        label = group_data['label'].iloc[0]

        new_row = list(avg_values) + [label]
        new_rows.append(new_row)
    else:
        print(f"⚠️  경고: {i}~{i+measure_count-1}행의 label이 일치하지 않음. 스킵.")

new_df = pd.DataFrame(new_rows, columns=df.columns)

print(f"\n✅ 그룹화 완료:")
print(f"그룹 크기: {measure_count}회씩")
print(f"생성된 샘플: {len(new_df)}개\n")
print(f"물질별 샘플 수:")
print(new_df['label'].value_counts())

# 저장
new_df.to_csv('plastic_data_averaged.csv', index=False)
print(f"\n💾 저장 완료: plastic_data_averaged.csv ({len(new_df)}행)")

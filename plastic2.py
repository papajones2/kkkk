import qwiic_as7265x
import time
import pandas as pd
import numpy as np

# ============================================================
# plastic2.py  — 기존 plastic.py 의 데이터 수집 방식 개선본
#
# [수정 1] 저장 방식 선택 추가
#          기존: 50회 측정 → 평균 1개만 저장 (plastic_data_average.csv)
#               → 종류당 10개 수집해도 CSV 에는 10행 밖에 없음
#               → 측정의 실제 변동성(분산)이 완전히 사라짐
#
#          수정 추가: 개별 측정값 전체 저장 옵션 (plastic_data_raw.csv)
#               → 50회 측정 → 50행으로 저장
#               → 실제 센서 노이즈·분산이 그대로 학습 데이터에 반영
#               → training2.py 는 이 파일을 우선 사용함
#
# [수정 2] 평균 저장 시에도 파일명을 average.csv 로 유지
#          → training2.py 와 파일명 연동
# ============================================================

CHANNELS = [
    '410nm', '435nm', '460nm', '485nm', '510nm', '535nm',
    '560nm', '585nm', '610nm', '645nm', '680nm', '705nm',
    '730nm', '760nm', '810nm', '860nm', '900nm', '940nm',
    'slope_IR_VIS'
]

LABEL_ORDER = ['PET', 'PP', 'PE', 'PS']


def read_sensor(sensor):
    sensor.take_measurements()
    return np.array([
        sensor.get_calibrated_a(), sensor.get_calibrated_b(), sensor.get_calibrated_c(),
        sensor.get_calibrated_d(), sensor.get_calibrated_e(), sensor.get_calibrated_f(),
        sensor.get_calibrated_g(), sensor.get_calibrated_h(), sensor.get_calibrated_r(),
        sensor.get_calibrated_i(), sensor.get_calibrated_s(), sensor.get_calibrated_j(),
        sensor.get_calibrated_t(), sensor.get_calibrated_u(), sensor.get_calibrated_v(),
        sensor.get_calibrated_w(), sensor.get_calibrated_k(), sensor.get_calibrated_l()
    ])


def load_baseline():
    try:
        baseline = pd.read_csv("baseline.csv", index_col=0)['value']
        print("✅ baseline 로드 성공")
        return baseline.values
    except FileNotFoundError:
        print("❌ baseline.csv 없음. 먼저 baseline.py 를 실행하세요.")
        exit()


def save_data(data_list, filename):
    df_new = pd.DataFrame(data_list, columns=CHANNELS + ['label'])
    try:
        df_existing = pd.read_csv(filename)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        print(f"기존 {len(df_existing)}개 + 새 {len(df_new)}개 = 총 {len(df_combined)}개")
    except FileNotFoundError:
        df_combined = df_new
        print(f"새 파일 생성. 총 {len(df_new)}개 데이터")

    df_combined['label'] = pd.Categorical(
        df_combined['label'], categories=LABEL_ORDER, ordered=True
    )
    df_combined = df_combined.sort_values(by='label', na_position='last').reset_index(drop=True)
    df_combined['label'] = df_combined['label'].astype(str)
    df_combined.to_csv(filename, index=False)
    print(f"💾 '{filename}' 저장 완료")


def compute_row(raw, baseline):
    safe_baseline = np.where(baseline == 0, 1e-10, baseline)
    ratio    = raw / safe_baseline
    ir_mean  = (ratio[16] + ratio[17]) / 2.0
    vis_mean = (ratio[0]  + ratio[1])  / 2.0
    slope    = ir_mean / vis_mean if vis_mean > 1e-10 else 0
    return list(ratio) + [slope]


if __name__ == "__main__":

    baseline = load_baseline()

    sensor = qwiic_as7265x.QwiicAS7265x(address=0x49)
    if sensor.begin() == False:
        print("❌ 센서 연결 실패")
        exit()
    print("✅ 센서 연결 성공!")

    sensor.enable_bulb(sensor.kLedUv)
    sensor.enable_bulb(sensor.kLedWhite)
    sensor.enable_bulb(sensor.kLedIr)

    label = input("\n플라스틱 종류 입력 : ").strip().upper()
    n     = int(input("측정 횟수 입력 (권장 50): ").strip())

    # ── [수정 1] 저장 방식 선택 ────────────────────────────────────
    print("\n저장 방식을 선택하세요:")
    print("  1 = 평균 1개만 저장  → plastic_data_average.csv  (기존 방식)")
    print("  2 = 개별 측정값 전체 → plastic_data_raw.csv      (권장: 실제 분산 반영)")
    save_mode = input("선택 (1 또는 2): ").strip()

    if save_mode == '2':
        target_file = "plastic_data_raw.csv"
        print(f"\n  → {n}개 개별 행으로 저장합니다.")
        print(f"  → training2.py 실행 시 이 파일을 자동으로 우선 사용합니다.\n")
    else:
        target_file = "plastic_data_average.csv"
        print(f"\n  → 평균값 1개 행으로 저장합니다.\n")

    print(f"--- {label} {n}회 측정 시작 ---")
    input("플라스틱 올리고 준비되면 Enter 누르세요...")

    measurements = []
    data_list    = []

    try:
        for i in range(1, n + 1):
            raw = read_sensor(sensor)
            measurements.append(raw)
            print(f"📊 {label} 측정 중... ({i}/{n})", end='\r')
            time.sleep(1)

        print(f"\n✅ {n}회 측정 완료!")

        if save_mode == '2':
            # ── [수정 1-A] 개별 측정값 전체를 각각 행으로 변환 ──────
            data_list = []
            for raw in measurements:
                row = compute_row(raw, baseline)
                data_list.append(row + [label])
            print(f"   → {len(data_list)}개 개별 행 준비 완료")
        else:
            # ── 기존 방식: n 회 평균을 1개 행으로 ───────────────────
            avg_raw = np.mean(measurements, axis=0)
            row     = compute_row(avg_raw, baseline)
            data_list = [row + [label]]

        save_data(data_list, target_file)

        sensor.disable_bulb(sensor.kLedUv)
        sensor.disable_bulb(sensor.kLedWhite)
        sensor.disable_bulb(sensor.kLedIr)

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        if data_list:
            print("현재까지 데이터 저장 시도...")
            save_data(data_list, target_file)

    except KeyboardInterrupt:
        print("\n🛑 강제 중단. 현재까지 저장합니다.")
        if data_list:
            save_data(data_list, target_file)

import qwiic_as7265x
import time
import pandas as pd
import numpy as np

# 💡 slope_IR_VIS가 제외된 18개 순수 파장 채널만 정의
CHANNELS = [
    '410nm', '435nm', '460nm', '485nm', '510nm', '535nm',
    '560nm', '585nm', '610nm', '645nm', '680nm', '705nm',
    '730nm', '760nm', '810nm', '860nm', '900nm', '940nm'
]

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
    except:
        print("❌ baseline.csv 없음. 먼저 baseline.py 실행하세요")
        exit()

def save_data(data_list, label, n):
    df_new = pd.DataFrame(data_list, columns=CHANNELS + ['label'])
    try:
        df_existing = pd.read_csv("plastic_data_average.csv")
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        print(f"기존 {len(df_existing)}개 + 새 {len(df_new)}개 = 총 {len(df_combined)}개")
    except:
        df_combined = df_new
        print(f"새 파일 생성. 총 {len(df_new)}개 데이터")

    # 🎯 PET -> PP -> PE -> PS 순서로 강제 정렬하는 로직
    custom_order = ['PET', 'PP', 'PE', 'PS']
    
    df_combined['label'] = pd.Categorical(
        df_combined['label'], 
        categories=custom_order, 
        ordered=True
    )
    
    df_combined = df_combined.sort_values(by='label', na_position='last').reset_index(drop=True)
    df_combined['label'] = df_combined['label'].astype(str)

    df_combined.to_csv("plastic_data_average.csv", index=False)
    print("💾 'plastic_data_average.csv' 정렬 및 저장 완료 (PET -> PP -> PE -> PS 순)")

if __name__ == "__main__":

    baseline = load_baseline()

    sensor = qwiic_as7265x.QwiicAS7265x(address=0x49)
    if sensor.begin() == False:
        print("❌ 센서 연결 실패")
        exit()
    print("✅ 센서 연결 성공!")

    sensor.enable_bulb(sensor.kLedUv)      # UV LED
    sensor.enable_bulb(sensor.kLedWhite)   # White LED
    sensor.enable_bulb(sensor.kLedIr)      # IR LED

    label = input("\n플라스틱 종류 입력 : ").strip().upper()
    n = int(input("측정 횟수 입력 (권장 50): ").strip())

    print(f"\n--- {label} {n}회 측정 시작 ---")
    input("플라스틱 올리고 준비되면 Enter 누르세요...")

    measurements = []
    data_list = []  # 예외 상황 안정성을 위한 초기화

    try:
        for i in range(1, n + 1):
            raw = read_sensor(sensor)
            measurements.append(raw)
            print(f"📊 {label} 측정 중... ({i}/{n})", end='\r')
            time.sleep(1)

        print(f"\n✅ {n}회 측정 완료!")

        # n회 측정값 평균
        avg_raw = np.mean(measurements, axis=0)

        # baseline 0 방지
        safe_baseline = np.where(baseline == 0, 1e-10, baseline)
        ratio = avg_raw / safe_baseline

        # 🎯 [변경 핵심] slope 관련 계산 및 데이터 가공 파트를 제거했습니다.
        row = list(ratio) + [label]
        data_list = [row]

        save_data(data_list, label, n)
        
        sensor.disable_bulb(sensor.kLedUv)      # UV LED
        sensor.disable_bulb(sensor.kLedWhite)   # White LED
        sensor.disable_bulb(sensor.kLedIr)      # IR LED

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        if data_list:
            print("현재까지 데이터 저장 시도...")
            save_data(data_list, label, len(data_list))

    except KeyboardInterrupt:
        print("\n🛑 강제 중단. 현재까지 저장합니다.")
        if data_list:
            save_data(data_list, label, len(data_list))
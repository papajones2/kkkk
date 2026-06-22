import qwiic_as7265x
import time
import pandas as pd
import numpy as np

CHANNELS = [
    '410nm', '435nm', '460nm', '485nm', '510nm', '535nm',
    '560nm', '585nm', '610nm', '645nm', '680nm', '705nm',
    '730nm', '760nm', '810nm', '860nm', '900nm', '940nm'
]

def read_sensor(sensor):
    sensor.take_measurements()
    return [
        sensor.get_calibrated_a(), sensor.get_calibrated_b(), sensor.get_calibrated_c(),
        sensor.get_calibrated_d(), sensor.get_calibrated_e(), sensor.get_calibrated_f(),
        sensor.get_calibrated_g(), sensor.get_calibrated_h(), sensor.get_calibrated_r(),
        sensor.get_calibrated_i(), sensor.get_calibrated_s(), sensor.get_calibrated_j(),
        sensor.get_calibrated_t(), sensor.get_calibrated_u(), sensor.get_calibrated_v(),
        sensor.get_calibrated_w(), sensor.get_calibrated_k(), sensor.get_calibrated_l()
    ]

if __name__ == "__main__":
    sensor = qwiic_as7265x.QwiicAS7265x(address=0x49)

    if sensor.begin() == False:
        print("❌ 센서 연결 실패")
        exit()
    print("✅ 센서 연결 성공!")

    sensor.enable_bulb(sensor.kLedUv)      # UV LED
    sensor.enable_bulb(sensor.kLedWhite)   # White LED
    sensor.enable_bulb(sensor.kLedIr)      # IR LED

    print("⏳ LED 워밍업 중... (5분)")
    print("   테스트 중이라면 Ctrl+C 누르면 5초로 단축됩니다")
    try:
        time.sleep(300)
    except KeyboardInterrupt:
        print("   워밍업 5초로 단축")
        time.sleep(5)

    print("\n--- baseline 측정 시작 ---")
    print("아크릴만 올려두세요 (플라스틱 없이)")
    input("준비되면 Enter 누르세요...")

    data_list = []

    try:
        for i in range(1, 11):
            row = read_sensor(sensor)
            data_list.append(row)
            print(f"📊 baseline 측정 중... ({i}/10)", end='\r')
            time.sleep(1)

        print("\n✅ baseline 10회 측정 완료!")

        df = pd.DataFrame(data_list, columns=CHANNELS)
        baseline_mean = df.mean()
        baseline_mean.to_csv("baseline.csv", header=['value'])

        print("\n📋 baseline 값:")
        for ch, val in baseline_mean.items():
            print(f"   {ch}: {val:.4f}")
        print("\n💾 'baseline.csv' 저장 완료")

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
    except KeyboardInterrupt:
        print("\n🛑 강제 중단")
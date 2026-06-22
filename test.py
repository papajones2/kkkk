import qwiic_as7265x
import time
import pandas as pd
import numpy as np
import pickle

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

def load_model():
    try:
        with open('model_svm.pkl', 'rb') as f:
            data = pickle.load(f)
        print("✅ 모델 로드 성공")
        return data['clf'], data['scaler']
    except:
        print("❌ model_svm.pkl 없음. 먼저 train.py 실행하세요")
        exit()

def load_baseline():
    try:
        baseline = pd.read_csv("baseline.csv", index_col=0)['value']
        print("✅ baseline 로드 성공")
        return baseline.values
    except:
        print("❌ baseline.csv 없음. 먼저 baseline.py 실행하세요")
        exit()

if __name__ == "__main__":

    clf, scaler = load_model()
    baseline = load_baseline()

    sensor = qwiic_as7265x.QwiicAS7265x(address=0x49)
    if sensor.begin() == False:
        print("❌ 센서 연결 실패")
        exit()
    print("✅ 센서 연결 성공!")

    sensor.enable_bulb(sensor.kLedUv)      # UV LED
    sensor.enable_bulb(sensor.kLedWhite)   # White LED
    sensor.enable_bulb(sensor.kLedIr)      # IR LED

    print("\n--- 예측 시작 (Ctrl+C로 종료) ---")

    while True:
        input("\n플라스틱 올리고 Enter 누르세요...")

        try:
            raw = read_sensor(sensor)
            safe_baseline = np.where(baseline == 0, 1e-10, baseline)
            ratio = raw / safe_baseline

            ir_mean = (ratio[16] + ratio[17]) / 2.0
            vis_mean = (ratio[0] + ratio[1]) / 2.0
            slope = ir_mean / vis_mean if vis_mean > 1e-10 else 0
            features = np.append(ratio, slope).reshape(1, -1)

            ratio_scaled = scaler.transform(features)
            prediction = clf.predict(ratio_scaled)[0]
            probability = clf.predict_proba(ratio_scaled)[0]

            print(f"\n🎯 예측 결과: {prediction}")
            print("📊 확률:")
            for label, prob in sorted(
                zip(clf.classes_, probability),
                key=lambda x: x[1],
                reverse=True
            ):
                bar = '█' * int(prob * 20)
                print(f"   {label:5s}: {bar:20s} {prob:.1%}")

        except Exception as e:
            print(f"❌ 오류: {e}")

        except KeyboardInterrupt:
            print("\n🛑 종료")
            break
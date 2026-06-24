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

    try:
        while True:
            input("\n플라스틱 올리고 Enter 누르세요...")

            try:
                raw = read_sensor(sensor)
                safe_baseline = np.where(baseline == 0, 1e-10, baseline)
                ratio = raw / safe_baseline

                # 🎯 [핵심 수정] 학습 모델 환경과 동일한 24개 피처 추출 파이프라인 구축
                
                # 안전한 분모 계산용 헬퍼 함수
                def safe_div(num, denom):
                    return num / (denom if denom > 1e-10 else 1e-10)

                # ① slope_IR_VIS 계산
                ir_mean = (ratio[16] + ratio[17]) / 2.0   # 900nm, 940nm
                vis_mean = (ratio[0] + ratio[1]) / 2.0     # 410nm, 435nm
                slope = safe_div(ir_mean, vis_mean)

                # ② 460nm 주변 차이 및 비율
                diff_460_435 = ratio[2] - ratio[1]
                ratio_460_435 = safe_div(ratio[2], ratio[1])

                # ③ 680nm 주변 차이 및 비율
                diff_680_645 = ratio[10] - ratio[9]
                ratio_680_645 = safe_div(ratio[10], ratio[9])

                # ④ 860nm 주변 비율
                ratio_860_810 = safe_div(ratio[15], ratio[14])

                # 💡 순수 18개 채널에 가공한 파생 변수 6개를 순서대로 붙여 정확히 24차원 데이터 생성
                derived_features = [
                    slope, 
                    diff_460_435, ratio_460_435, 
                    diff_680_645, ratio_680_645, 
                    ratio_860_810
                ]
                
                # 차원 정렬 (1, 24) 형태로 스케일러에 전달
                features = np.append(ratio, derived_features).reshape(1, -1)

                # StandardScaler 및 모델 예측 진행
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
                print(f"❌ 내부 오류: {e}")

    except KeyboardInterrupt:
        print("\n🛑 예측 프로그램을 종료합니다.")
    finally:
        # 안전한 LED 종료 처리
        sensor.disable_bulb(sensor.kLedUv)      
        sensor.disable_bulb(sensor.kLedWhite)   
        sensor.disable_bulb(sensor.kLedIr)      
        print("💡 센서 LED 안전 차단 완료.")
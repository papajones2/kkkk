import qwiic_as7265x
import time
import pandas as pd
import numpy as np
import pickle

# ============================================================
# test2.py  — 기존 test.py 의 2가지 문제 수정본
#
# [수정 1] 단일 측정 → N_MEASURE 회 평균 측정
#          학습 데이터(plastic.py) 는 50회 평균값인데
#          기존 test.py 는 1회 단일값으로 예측 → 노이즈 차이 발생
#          수정: 예측 시에도 여러 번 측정 후 평균 사용
#
# [수정 2] 신뢰도 임계값 추가 (Open Set 문제 대응)
#          기존: 학습 안 된 새 종류를 넣어도 무조건 PET/PP/PE/PS 출력
#          수정: 최고 확률이 CONFIDENCE_THRESHOLD 미만이면
#               "알 수 없는 종류" 경고 출력
#
# [수정 3] Pipeline 모델 로드
#          training2.py 가 scaler+clf 를 Pipeline 으로 저장하므로
#          별도 scaler.transform 불필요 — pipeline.predict 만 호출
# ============================================================

N_MEASURE            = 5     # 예측 1회당 측정 횟수 (학습 데이터의 50회보다 적지만 노이즈 평균화)
CONFIDENCE_THRESHOLD = 0.60  # 최고 확률이 이 미만이면 "미확인 종류" 경고


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
        with open('model_svm3.pkl', 'rb') as f:
            pipeline = pickle.load(f)
        print("✅ 모델 로드 성공 (model_svm3.pkl)")
        return pipeline
    except FileNotFoundError:
        print("❌ model_svm3.pkl 없음. 먼저 training_v3.py 를 실행하세요.")
        exit()


def load_baseline():
    try:
        baseline = pd.read_csv("baseline.csv", index_col=0)['value']
        print("✅ baseline 로드 성공")
        return baseline.values
    except FileNotFoundError:
        print("❌ baseline.csv 없음. 먼저 baseline.py 를 실행하세요.")
        exit()


if __name__ == "__main__":

    pipeline = load_model()
    baseline = load_baseline()

    sensor = qwiic_as7265x.QwiicAS7265x(address=0x49)
    if sensor.begin() == False:
        print("❌ 센서 연결 실패")
        exit()
    print("✅ 센서 연결 성공!")

    sensor.enable_bulb(sensor.kLedUv)
    sensor.enable_bulb(sensor.kLedWhite)
    sensor.enable_bulb(sensor.kLedIr)

    known_classes = list(pipeline.classes_)
    print(f"\n학습된 플라스틱 종류: {known_classes}")
    print(f"신뢰도 임계값: {CONFIDENCE_THRESHOLD:.0%}  (미만이면 미확인 종류 경고)")
    print(f"측정 횟수/예측: {N_MEASURE}회 평균")
    print("\n--- 예측 시작 (Ctrl+C 로 종료) ---")

    while True:
        input("\n플라스틱 올리고 Enter 누르세요...")

        try:
            # ── [수정 1] N_MEASURE 회 측정 후 평균 ──────────────────
            print(f"📊 {N_MEASURE}회 측정 중", end='', flush=True)
            measurements = []
            for _ in range(N_MEASURE):
                measurements.append(read_sensor(sensor))
                time.sleep(0.3)
                print(".", end='', flush=True)
            avg_raw = np.mean(measurements, axis=0)
            print(" 완료!")

            # ── baseline 비율 계산 ───────────────────────────────────
            safe_baseline = np.where(baseline == 0, 1e-10, baseline)
            ratio    = avg_raw / safe_baseline
            ir_mean  = (ratio[16] + ratio[17]) / 2.0
            vis_mean = (ratio[0]  + ratio[1])  / 2.0
            slope    = ir_mean / vis_mean if vis_mean > 1e-10 else 0
            features = np.append(ratio, slope).reshape(1, -1)

            # ── [수정 3] Pipeline 으로 예측 (scaler 자동 적용) ───────
            prediction  = pipeline.predict(features)[0]
            probability = pipeline.predict_proba(features)[0]
            max_prob    = max(probability)

            # ── [수정 2] 신뢰도 임계값으로 미확인 종류 감지 ─────────
            print()
            if max_prob < CONFIDENCE_THRESHOLD:
                print(f"⚠️  신뢰도 부족 ({max_prob:.1%})")
                print(f"   → 학습된 종류({', '.join(known_classes)}) 가 아닐 수 있습니다.")
                print(f"   → 가장 가까운 추정: {prediction}  (신뢰하지 마세요)")
            else:
                print(f"🎯 예측 결과: {prediction}  (신뢰도: {max_prob:.1%})")

            print("📊 전체 확률:")
            sorted_results = sorted(zip(pipeline.classes_, probability), key=lambda x: x[1], reverse=True)
            for lbl, prob in sorted_results:
                bar = '█' * int(prob * 20)
                print(f"   {lbl:5s}: {bar:20s} {prob:.1%}")

        except Exception as e:
            print(f"❌ 오류: {e}")

        except KeyboardInterrupt:
            print("\n🛑 종료합니다.")
            sensor.disable_bulb(sensor.kLedUv)
            sensor.disable_bulb(sensor.kLedWhite)
            sensor.disable_bulb(sensor.kLedIr)
            break

"""
server.py  —  Flask REST API 서버 (라즈베리파이 실행용)

실행: python server.py
앱에서 POST http://<라즈베리파이IP>:5000/predict 로 요청
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import threading
import pickle
import numpy as np
import pandas as pd
from flask import Flask, jsonify

app = Flask(__name__)
sensor_lock = threading.Lock()  # 동시 요청 시 센서 충돌 방지

# ── 시작 시 1회 로드 ──────────────────────────────────────────────────────
try:
    import qwiic_as7265x
    sensor = qwiic_as7265x.QwiicAS7265x(address=0x49)
    if not sensor.begin():
        raise RuntimeError("센서 연결 실패")
    sensor.enable_bulb(sensor.kLedUv)
    sensor.enable_bulb(sensor.kLedWhite)
    sensor.enable_bulb(sensor.kLedIr)
    SENSOR_OK = True
    print("✅ 센서 연결 성공")
except Exception as e:
    SENSOR_OK = False
    sensor = None
    print(f"⚠️  센서 없음: {e}")

MODEL_FILE = 'model_svm3.pkl'
try:
    with open(MODEL_FILE, 'rb') as f:
        pipeline = pickle.load(f)
    print(f"✅ 모델 로드: {MODEL_FILE}")
    print(f"   클래스: {list(pipeline.classes_)}")
except FileNotFoundError:
    print(f"❌ {MODEL_FILE} 없음")
    sys.exit(1)

try:
    baseline = pd.read_csv('baseline.csv', index_col=0)['value'].values
    print("✅ baseline 로드 완료")
except FileNotFoundError:
    print("❌ baseline.csv 없음 — baseline.py 먼저 실행하세요")
    sys.exit(1)


# ── 센서 읽기 ─────────────────────────────────────────────────────────────
def read_sensor():
    sensor.take_measurements()
    return np.array([
        sensor.get_calibrated_a(), sensor.get_calibrated_b(), sensor.get_calibrated_c(),
        sensor.get_calibrated_d(), sensor.get_calibrated_e(), sensor.get_calibrated_f(),
        sensor.get_calibrated_g(), sensor.get_calibrated_h(), sensor.get_calibrated_r(),
        sensor.get_calibrated_i(), sensor.get_calibrated_s(), sensor.get_calibrated_j(),
        sensor.get_calibrated_t(), sensor.get_calibrated_u(), sensor.get_calibrated_v(),
        sensor.get_calibrated_w(), sensor.get_calibrated_k(), sensor.get_calibrated_l()
    ])


# ── 엔드포인트 ────────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    """앱이 서버 연결 확인용으로 사용"""
    return jsonify({
        'status': 'ok',
        'sensor': SENSOR_OK,
        'model': MODEL_FILE
    })


@app.route('/predict', methods=['POST'])
def predict():
    """측정 요청 — 센서 읽고 예측 결과 JSON 반환"""
    if not SENSOR_OK:
        return jsonify({'error': '센서가 연결되지 않았습니다'}), 503

    with sensor_lock:
        try:
            raw = read_sensor()
            safe_baseline = np.where(baseline == 0, 1e-10, baseline)
            ratio = raw / safe_baseline  # 18채널

            # 19번째 파생 피처: slope_IR_VIS = mean(900nm,940nm) / mean(410nm,435nm)
            # ratio 인덱스: 0=410nm, 1=435nm, 16=900nm, 17=940nm
            ir_mean  = (ratio[16] + ratio[17]) / 2.0
            vis_mean = (ratio[0]  + ratio[1])  / 2.0
            slope_ir_vis = ir_mean / (vis_mean if vis_mean != 0 else 1e-10)
            X = np.append(ratio, slope_ir_vis).reshape(1, -1)

            prediction = str(pipeline.predict(X)[0])
            probs = pipeline.predict_proba(X)[0]

            probabilities = {
                str(label): round(float(prob), 4)
                for label, prob in sorted(
                    zip(pipeline.classes_, probs),
                    key=lambda x: x[1], reverse=True
                )
            }

            print(f"📦 예측: {prediction}  {probabilities}")
            return jsonify({
                'prediction': prediction,
                'probabilities': probabilities
            })

        except Exception as e:
            print(f"❌ 예측 오류: {e}")
            return jsonify({'error': str(e)}), 500


# ── 서버 시작 ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"\n🚀 서버 시작!")
    print(f"   안드로이드 앱 IP 입력창에 입력: {local_ip}")
    print(f"   (또는 라즈베리파이에서 ifconfig 로 확인)")
    print(f"   종료: Ctrl+C\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)

import sys
sys.stdout.reconfigure(encoding='utf-8')

import threading
import numpy as np
import pandas as pd
from flask import Flask, jsonify

app = Flask(__name__)
sensor_lock = threading.Lock()  # 동시 요청 시 센서 충돌 방지

# ── 시작 시 1회 센서 및 베이스라인 로드 ───────────────────────────────────
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
    print(f"⚠️ 센서 없음: {e}")

try:
    baseline = pd.read_csv('baseline.csv', index_col=0)['value'].values
    print("✅ baseline 로드 완료")
except FileNotFoundError:
    print("❌ baseline.csv 없음 — 라즈베리파이에 이 파일이 있어야 합니다.")
    sys.exit(1)

# ── 센서 읽기 함수 ────────────────────────────────────────────────────────
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

# ── 노트북 서버 전용 엔드포인트 ──────────────────────────────────────────
@app.route('/get-raw', methods=['GET'])
def get_raw():
    """노트북 서버가 호출하면 센서 로우 데이터와 베이스라인을 리스트로 반환"""
    if not SENSOR_OK:
        return jsonify({'error': '센서가 연결되지 않았습니다'}), 503

    with sensor_lock:
        try:
            raw = read_sensor()
            safe_baseline = np.where(baseline == 0, 1e-10, baseline)
            
            # numpy 배열을 무선(JSON)으로 전송하기 위해 list로 변환
            return jsonify({
                'raw_data': raw.tolist(),
                'safe_baseline': safe_baseline.tolist()
            })
        except Exception as e:
            print(f"❌ 센서 읽기 오류: {e}")
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n🚀 라즈베리파이 무선 센서 노드 시작!")
    print("   노트북의 요청을 대기합니다... (Port: 5000)\n")
    # 호스트를 0.0.0.0으로 열어야 노트북이 무선으로 찌를 수 있습니다.
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)
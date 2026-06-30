"""
laptop_server.py  —  노트북 CMD 에서 실행하는 Flask 서버

실행: python laptop_server.py
앱에서 POST http://<노트북IP>:5000/predict 로 요청

필요 패키지 (노트북):
    pip install flask paramiko numpy pandas scikit-learn
필요 파일 (이 파일과 같은 폴더):
    model_svm3.pkl
    baseline.csv
"""
import sys
import json
import pickle
import threading
import numpy as np
import pandas as pd
import paramiko
from flask import Flask, jsonify

app = Flask(__name__)
sensor_lock = threading.Lock()

# ── 라즈베리파이 SSH 설정 ─────────────────────────────────────────────────
PI_HOST     = "172.20.10.13"   # ← 라즈베리파이 IP 로 변경 (ifconfig 로 확인)
PI_USER     = "pi"
PI_PASSWORD = "1234"      # ← 라즈베리파이 비밀번호로 변경
PI_SCRIPT   = "/home/pi/test.py"

# ── 시작 시 1회 로드 ──────────────────────────────────────────────────────
MODEL_FILE = 'model_svm3.pkl'
try:
    with open(MODEL_FILE, 'rb') as f:
        pipeline = pickle.load(f)
    print(f"✅ 모델 로드: {MODEL_FILE}")
    print(f"   클래스: {list(pipeline.classes_)}")
except FileNotFoundError:
    print(f"❌ {MODEL_FILE} 없음 — 노트북 폴더에 모델 파일을 넣으세요")
    sys.exit(1)

try:
    baseline = pd.read_csv('baseline.csv', index_col=0)['value'].values
    print("✅ baseline 로드 완료")
except FileNotFoundError:
    print("❌ baseline.csv 없음 — 노트북 폴더에 파일을 넣으세요")
    sys.exit(1)


# ── 라즈베리파이에 SSH 접속해서 센서값 가져오기 ───────────────────────────
def get_sensor_via_ssh() -> np.ndarray:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(PI_HOST, username=PI_USER, password=PI_PASSWORD, timeout=10)

    _, stdout, stderr = ssh.exec_command(f'python3 {PI_SCRIPT}')
    stdout.channel.recv_exit_status()  # 실행 완료 대기
    output = stdout.read().decode('utf-8').strip()
    err    = stderr.read().decode('utf-8').strip()
    ssh.close()

    if not output:
        raise RuntimeError(f"Pi 응답 없음: {err}")

    data = json.loads(output)
    if 'error' in data:
        raise RuntimeError(f"Pi 오류: {data['error']}")

    CHANNELS = [
        '410nm', '435nm', '460nm', '485nm', '510nm', '535nm',
        '560nm', '585nm', '610nm', '645nm', '680nm', '705nm',
        '730nm', '760nm', '810nm', '860nm', '900nm', '940nm'
    ]
    return np.array([data[ch] for ch in CHANNELS])


# ── 엔드포인트 ────────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'pi_host': PI_HOST,
        'model': MODEL_FILE
    })
    


@app.route('/predict', methods=['POST'])
def predict():
    with sensor_lock:
        try:
            raw = get_sensor_via_ssh()

            safe_baseline = np.where(baseline == 0, 1e-10, baseline)
            ratio = raw / safe_baseline  # 18채널

            # 19번째 파생 피처: slope_IR_VIS
            ir_mean      = (ratio[16] + ratio[17]) / 2.0
            vis_mean     = (ratio[0]  + ratio[1])  / 2.0
            slope_ir_vis = ir_mean / (vis_mean if vis_mean != 0 else 1e-10)
            X = np.append(ratio, slope_ir_vis).reshape(1, -1)

            prediction = str(pipeline.predict(X)[0])
            probs       = pipeline.predict_proba(X)[0]

            probabilities = {
                str(label): round(float(prob), 4)
                for label, prob in sorted(
                    zip(pipeline.classes_, probs),
                    key=lambda x: x[1], reverse=True
                )
            }

            print(f"📦 예측: {prediction}  {probabilities}")
            return jsonify({
                'prediction':   prediction,
                'probabilities': probabilities
            })

        except Exception as e:
            print(f"❌ 오류: {e}")
            return jsonify({'error': str(e)}), 500


# ── 서버 시작 ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import socket
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "확인 필요 (ipconfig)"

    print(f"\n🚀 노트북 서버 시작!")
    print(f"   라즈베리파이 IP 설정: {PI_HOST}")
    print(f"   안드로이드 앱 IP 입력창에 입력: {local_ip}")
    print(f"   종료: Ctrl+C\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)

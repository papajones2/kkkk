"""
test.py  —  라즈베리파이 전용 (센서 측정만)

노트북 server.py 가 SSH 로 실행:
    python3 /home/pi/test.py
결과를 JSON 한 줄로 stdout 출력 후 종료.
모델/baseline 없음 — 모든 AI 처리는 노트북에서 수행.
"""
import sys
import json
import qwiic_as7265x

CHANNELS = [
    '410nm', '435nm', '460nm', '485nm', '510nm', '535nm',
    '560nm', '585nm', '610nm', '645nm', '680nm', '705nm',
    '730nm', '760nm', '810nm', '860nm', '900nm', '940nm'
]

def main():
    sensor = qwiic_as7265x.QwiicAS7265x(address=0x49)
    if not sensor.begin():
        print(json.dumps({"error": "센서 연결 실패"}), flush=True)
        sys.exit(1)

    sensor.enable_bulb(sensor.kLedUv)
    sensor.enable_bulb(sensor.kLedWhite)
    sensor.enable_bulb(sensor.kLedIr)

    sensor.take_measurements()

    raw = [
        sensor.get_calibrated_a(), sensor.get_calibrated_b(), sensor.get_calibrated_c(),
        sensor.get_calibrated_d(), sensor.get_calibrated_e(), sensor.get_calibrated_f(),
        sensor.get_calibrated_g(), sensor.get_calibrated_h(), sensor.get_calibrated_r(),
        sensor.get_calibrated_i(), sensor.get_calibrated_s(), sensor.get_calibrated_j(),
        sensor.get_calibrated_t(), sensor.get_calibrated_u(), sensor.get_calibrated_v(),
        sensor.get_calibrated_w(), sensor.get_calibrated_k(), sensor.get_calibrated_l()
    ]

    sensor.disable_bulb(sensor.kLedUv)
    sensor.disable_bulb(sensor.kLedWhite)
    sensor.disable_bulb(sensor.kLedIr)

    print(json.dumps({ch: val for ch, val in zip(CHANNELS, raw)}), flush=True)

if __name__ == '__main__':
    main()

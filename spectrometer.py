import qwiic_as7265x
import time
import pandas as pd

# 센서 객체 생성 (주소 0x49)
sensor = qwiic_as7265x.QwiicAS7265x(address=0x49)

print("--- 10회 자동 측정 모드 시작 ---")

# 센서 초기화 확인
if sensor.begin() == False:
    print("❌ 센서 하드웨어 응답 없음! 연결을 확인하세요.")
    exit()

print("✅ 센서 연결 성공!")

sensor.enable_bulb_uv()     # UV LED
sensor.enable_bulb_white()  # 가시광선 LED
sensor.enable_bulb_ir()     # IR LED

data_list = []
max_measurements = 10  # 측정 횟수 설정
columns = ['410nm', '435nm', '460nm', '485nm', '510nm', '535nm', 
           '560nm', '585nm', '610nm', '645nm', '705nm', '760nm', 
           '810nm', '860nm', '900nm', '940nm', '1000nm', '1050nm']

try:
    for i in range(1, max_measurements + 1):
        # 1. 측정 지시
        sensor.take_measurements()
        
        # 2. 데이터 수집 (안전한 개별 채널 읽기 방식)
        row = [
            sensor.get_calibrated_a(), sensor.get_calibrated_b(), sensor.get_calibrated_c(),
            sensor.get_calibrated_d(), sensor.get_calibrated_e(), sensor.get_calibrated_f(),
            sensor.get_calibrated_g(), sensor.get_calibrated_h(), sensor.get_calibrated_i(),
            sensor.get_calibrated_j(), sensor.get_calibrated_k(), sensor.get_calibrated_l(),
            sensor.get_calibrated_r(), sensor.get_calibrated_s(), sensor.get_calibrated_t(),
            sensor.get_calibrated_u(), sensor.get_calibrated_v(), sensor.get_calibrated_w()
        ]
        
        data_list.append(row)
        print(f"📊 측정 중... ({i}/{max_measurements})", end='\r')
        
        time.sleep(1) # 1초 간격으로 측정

    # 10번 측정이 끝나면 저장 단계로 이동
    print("\n\n✅ 10회 측정 완료!")
    df = pd.DataFrame(data_list, columns=columns)
    df.to_csv("result.csv", index=False)
    print("💾 'result.csv' 파일로 저장이 완료되었습니다.")

except Exception as e:
    print(f"\n❌ 오류 발생: {e}")

except KeyboardInterrupt:
    print("\n🛑 사용자가 강제 중단했습니다. 현재까지만 저장합니다.")
    if data_list:
        df = pd.DataFrame(data_list, columns=columns)
        df.to_csv("result.csv", index=False)
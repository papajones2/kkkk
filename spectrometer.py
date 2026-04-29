import qwiic_as7265x
import time
import pandas as pd

# 센서 연결 확인
sensor = qwiic_as7265x.QwiicAS7265x()

if sensor.begin() == False:
    print("센서가 연결되지 않았습니다. 선을 확인하세요!")
    exit()

print("측정을 시작합니다... (중단하려면 Ctrl + C)")

data_list = []

try:
    while True:
        sensor.take_measurements()
        # 모든 파장(18개) 데이터 읽기
        data = sensor.get_all_color_data()
        data_list.append(data)
        
        print(f"현재 {len(data_list)}개 데이터 수집됨...")
        time.sleep(1) # 1초 간격

except KeyboardInterrupt:
    # 엑셀(CSV) 파일로 저장
    df = pd.DataFrame(data_list)
    df.to_csv("result.csv", index=False)
    print("\n측정 종료! result.csv 파일로 저장되었습니다.")
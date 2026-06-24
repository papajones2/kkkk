import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pickle

# 💡 원본 CSV 파일에 들어있는 순수 18개 파장 채널
RAW_CHANNELS = [
    '410nm', '435nm', '460nm', '485nm', '510nm', '535nm',
    '560nm', '585nm', '610nm', '645nm', '680nm', '705nm',
    '730nm', '760nm', '810nm', '860nm', '900nm', '940nm'
]

def train():
    # 데이터 로드
    try:
        df = pd.read_csv("plastic_data_average2.csv")
        print(f"✅ 데이터 로드 성공: 총 {len(df)}개")
        print(f"   종류별 개수:\n{df['label'].value_counts()}")
    except:
        print("❌ plastic_data_average2.csv 없음. 먼저 measure.py와 augmentation.py를 실행하세요")
        exit()

    # 데이터 확인
    if len(df['label'].unique()) < 2:
        print("❌ 최소 2종류 이상 플라스틱 데이터 필요")
        exit()

    # 🎯 [핵심 변경 1] 로드된 18개 파장 데이터를 기반으로 학습 직전 피처 엔지니어링 수행
    print("\n🛠️ 파생 특징(slope_IR_VIS 및 핵심 파장 비율/차이) 계산 중...")
    
    # 안전한 나눗셈을 위한 분모 0 방지 함수
    def safe_div(num, denom):
        return num / np.where(denom == 0, 1e-10, denom)

    # ① 기존 measure에서 하던 slope_IR_VIS를 여기서 계산하여 추가
    ir_mean = (df['900nm'] + df['940nm']) / 2.0
    vis_mean = (df['410nm'] + df['435nm']) / 2.0
    df['slope_IR_VIS'] = safe_div(ir_mean, vis_mean)

    # ② 460nm 피크 주변부의 상대적 변화량 및 비율
    df['diff_460_435'] = df['460nm'] - df['435nm']
    df['ratio_460_435'] = safe_div(df['460nm'], df['435nm'])

    # ③ 680nm 피크 주변부의 상대적 변화량 및 비율
    df['diff_680_645'] = df['680nm'] - df['645nm']
    df['ratio_680_645'] = safe_div(df['680nm'], df['645nm'])

    # ④ 860nm 피크의 증폭 비율
    df['ratio_860_810'] = safe_div(df['860nm'], df['810nm'])

    # 💡 최종 학습에 사용될 전체 피처 목록 정의 (총 24개)
    ALL_FEATURES = RAW_CHANNELS + [
        'slope_IR_VIS',
        'diff_460_435', 'ratio_460_435', 
        'diff_680_645', 'ratio_680_645', 
        'ratio_860_810'
    ]

    X = df[ALL_FEATURES].values
    y = df['label'].values

    # 정규화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA 시각화
    print("\n📊 PCA 시각화 중...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    for label in np.unique(y):
        mask = y == label
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=label, alpha=0.6)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.title('플라스틱 종류별 분포 (피처 추가 후)')
    plt.tight_layout()
    plt.savefig('pca_plot.png')
    plt.show()
    print(f"   PC1+PC2 설명 분산: {pca.explained_variance_ratio_.sum():.1%}")

    # 학습/테스트 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # 분류기 학습 (일반화 성능 향상을 위해 C값 최적화)
    print("\n🤖 최적화된 SVM 모델 학습 중...")
    clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True)
    clf.fit(X_train, y_train)

    # 성능 확인
    y_pred = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)

    train_accuracy = (y_pred_train == y_train).mean() * 100
    test_accuracy = (y_pred == y_test).mean() * 100

    print("\n" + "="*50)
    print("📊 최종 모델 정확도 (Accuracy)")
    print("="*50)
    print(f"🏋️   Train Accuracy: {train_accuracy:.2f}%")
    print(f"🧪 Test Accuracy:  {test_accuracy:.2f}%")
    print("="*50)
    print("\n📋 상세 분류 성능:")
    print(classification_report(y_test, y_pred))

    # 교차검증
    scores = cross_val_score(clf, X_scaled, y, cv=5)
    print(f"교차검증 정확도: {scores.mean():.2f} ± {scores.std():.2f}")

    # 📊 채널 및 파생 피처 중요도 그래프 생성
    print("\n📊 피처별 중요도 그래프 생성 중...")
    plt.figure(figsize=(12, 8))
    
    temp_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    temp_rf.fit(X_train, y_train)
    
    importances = pd.Series(temp_rf.feature_importances_, index=ALL_FEATURES)
    importances.sort_values().plot(kind='barh', color='skyblue')
    
    plt.title('피처별 중요도 (slope 및 추가 피처 영향도 확인)')
    plt.xlabel('중요도 (Importance)')
    plt.tight_layout()
    plt.savefig('importance_plot.png')
    plt.show()

    # 모델 저장 (실제 고성능 분류기인 SVM 객체와 스케일러 저장)
    with open('model_svm.pkl', 'wb') as f:
        pickle.dump({'clf': clf, 'scaler': scaler}, f)
    print("\n💾 모델 저장 완료 (model_svm.pkl)")

    return clf, scaler

if __name__ == "__main__":
    train()
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

CHANNELS = [
    '410nm', '435nm', '460nm', '485nm', '510nm', '535nm',
    '560nm', '585nm', '610nm', '645nm', '680nm', '705nm',
    '730nm', '760nm', '810nm', '860nm', '900nm', '940nm',
    'slope_IR_VIS'
]

def train():
    # 데이터 로드
    try:
        df = pd.read_csv("plastic_data_average2.csv")
        print(f"✅ 데이터 로드 성공: 총 {len(df)}개")
        print(f"   종류별 개수:\n{df['label'].value_counts()}")
    except:
        print("❌ plastic_data_average2.csv 없음. 먼저 measure.py 실행하세요")
        exit()

    # 데이터 확인
    if len(df['label'].unique()) < 2:
        print("❌ 최소 2종류 이상 플라스틱 데이터 필요")
        exit()

    X = df[CHANNELS].values
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
    plt.title('플라스틱 종류별 분포\n(덩어리가 잘 분리될수록 분류 정확도 높음)')
    plt.tight_layout()
    plt.savefig('pca_plot.png')
    plt.show()
    print(f"   PC1+PC2 설명 분산: {pca.explained_variance_ratio_.sum():.1%}")

    # 학습/테스트 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # 분류기 학습 (SVM 모델 사용)
    print("\n🤖 SVM 모델 학습 중...")
    clf = SVC(kernel='rbf', C=1.0, random_state=42, probability=True)
    clf.fit(X_train, y_train)

    # 성능 확인
    y_pred = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)

    train_accuracy = (y_pred_train == y_train).mean() * 100
    test_accuracy = (y_pred == y_test).mean() * 100

    print("\n" + "="*50)
    print("📊 정확도 (Accuracy)")
    print("="*50)
    print(f"🏋️  Train Accuracy: {train_accuracy:.2f}%")
    print(f"🧪 Test Accuracy:  {test_accuracy:.2f}%")
    print("="*50)
    print("\n📋 상세 분류 성능:")
    print(classification_report(y_test, y_pred))

    # 교차검증
    scores = cross_val_score(clf, X_scaled, y, cv=5)
    print(f"교차검증 정확도: {scores.mean():.2f} ± {scores.std():.2f}")

    # 📊 채널별 중요도 그래프 생성 파트 (에러 해결 우회로 부품 추가)
    print("\n📊 파장별 중요도 그래프 생성 중...")
    plt.figure(figsize=(10, 6))
    
    # SVM에는 없으므로 중요도 추출을 위해 가볍게 랜덤 포레스트를 임시 학습시킵니다.
    temp_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    temp_rf.fit(X_train, y_train)
    
    # 임시 모델에서 중요도를 뽑아 그래프를 그립니다.
    importances = pd.Series(temp_rf.feature_importances_, index=CHANNELS)
    importances.sort_values().plot(kind='barh', color='skyblue')
    
    plt.title('채널별 중요도 (높을수록 플라스틱 분류에 중요한 파장)')
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
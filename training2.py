import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pickle

# ============================================================
# training2.py  — 기존 training.py 의 3가지 문제 수정본
#
# [수정 1] 스케일러 데이터 누출 제거
#          기존: 전체 데이터로 scaler.fit → train/test 분리
#          수정: train/test 분리 후 train 데이터로만 scaler.fit
#               → Pipeline 으로 묶어서 교차검증도 안전하게
#
# [수정 2] 증강 데이터 누출 제거
#          기존: 이미 증강된 average2.csv 를 통째로 train_test_split
#               → 같은 원본의 증강본이 train/test 양쪽에 들어가 정확도 부풀림
#          수정: 원본(average.csv) 기준으로 분리 먼저,
#               train 부분만 증강해서 학습
#               (plastic_data_raw.csv 가 있으면 그걸 우선 사용)
#
# [수정 3] 교차검증 누출 제거
#          기존: 이미 스케일된 X_scaled 로 cross_val_score
#          수정: Pipeline 으로 cross_val_score → 매 fold 마다 scaler 재fit
# ============================================================

CHANNELS = [
    '410nm', '435nm', '460nm', '485nm', '510nm', '535nm',
    '560nm', '585nm', '610nm', '645nm', '680nm', '705nm',
    '730nm', '760nm', '810nm', '860nm', '900nm', '940nm',
    'slope_IR_VIS'
]


def augment_train(X, y, n_copies=4, noise_std=0.02):
    """
    훈련 데이터만 증강한다.
    noise_std: 원본 비율값 기준 약 ±2% 노이즈 — 센서 실측 변동 범위 내
    """
    np.random.seed(42)
    X_list = [X]
    y_list = [y]
    for _ in range(n_copies):
        noise = np.random.normal(0, noise_std, X.shape)
        X_list.append(X + noise)
        y_list.append(y)
    return np.vstack(X_list), np.concatenate(y_list)


def load_data():
    """
    plastic_data_raw.csv (개별 측정값) 우선 사용.
    없으면 plastic_data_average.csv (평균값 원본) 사용.
    average2.csv 는 증강 누출 문제가 있으므로 직접 사용하지 않음.
    """
    for fname in ["plastic_data_raw.csv", "plastic_data_average.csv"]:
        try:
            df = pd.read_csv(fname)
            print(f"✅ 데이터 로드: {fname}  ({len(df)}개)")
            print(f"   종류별 개수:\n{df['label'].value_counts()}\n")
            return df, fname
        except FileNotFoundError:
            continue

    print("❌ plastic_data_raw.csv / plastic_data_average.csv 없음.")
    print("   plastic2.py 로 데이터를 먼저 수집하세요.")
    exit()


def train():
    df, source_file = load_data()

    if len(df['label'].unique()) < 2:
        print("❌ 최소 2종류 이상 플라스틱 데이터 필요")
        exit()

    X = df[CHANNELS].values
    y = df['label'].values

    # ── [수정 2] 원본 기준으로 먼저 분리 ────────────────────────────
    X_train_orig, X_test, y_train_orig, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"📊 분리 완료 (증강 전 원본 기준)")
    print(f"   Train 원본: {len(X_train_orig)}개  /  Test(원본만): {len(X_test)}개")

    # raw 파일(개별 측정)이면 증강 불필요, average 파일이면 증강 적용
    if "raw" in source_file:
        X_train, y_train = X_train_orig, y_train_orig
        print(f"   → raw 파일 사용 중: 실제 분산이 포함되어 있어 증강 생략\n")
    else:
        X_train, y_train = augment_train(X_train_orig, y_train_orig, n_copies=4)
        print(f"   → average 파일 사용 중: train 부분만 4배 증강 → {len(X_train)}개\n")

    # ── PCA 시각화 ────────────────────────────────────────────────
    print("📊 PCA 시각화 중...")
    X_all    = np.vstack([X_train, X_test])
    y_all    = np.concatenate([y_train, y_test])
    vis_sc   = StandardScaler()
    X_all_sc = vis_sc.fit_transform(X_all)

    pca   = PCA(n_components=2)
    X_pca = pca.fit_transform(X_all_sc)

    plt.figure(figsize=(8, 6))
    for lbl in np.unique(y_all):
        mask = y_all == lbl
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=lbl, alpha=0.5)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.legend()
    plt.title('PCA 분포 (증강 포함)\n덩어리가 잘 분리될수록 분류 정확도 높음')
    plt.tight_layout()
    plt.savefig('pca_plot2.png')
    plt.show()
    print(f"   PC1+PC2 설명 분산: {pca.explained_variance_ratio_.sum():.1%}\n")

    # ── [수정 1] Pipeline 으로 scaler + clf 묶기 ─────────────────
    # Pipeline 은 fit 시 train 데이터만 보고 scaler 를 fit 함
    print("🤖 SVM 모델 학습 중...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    SVC(kernel='rbf', C=1.0, random_state=42, probability=True))
    ])
    pipeline.fit(X_train, y_train)

    # ── 성능 확인 ─────────────────────────────────────────────────
    y_pred_train = pipeline.predict(X_train)
    y_pred_test  = pipeline.predict(X_test)

    train_acc = (y_pred_train == y_train).mean() * 100
    test_acc  = (y_pred_test  == y_test ).mean() * 100

    print("\n" + "="*55)
    print("📊 정확도 (Accuracy)")
    print("="*55)
    print(f"🏋️  Train Accuracy : {train_acc:.2f}%  (증강 데이터 기준, 참고용)")
    print(f"🧪 Test Accuracy  : {test_acc:.2f}%  ← 이 값이 실제 성능 지표")
    print("="*55)
    print("\n📋 상세 분류 성능 (Test set — 원본 데이터):")
    print(classification_report(y_test, y_pred_test))

    # ── [수정 3] Pipeline 으로 교차검증 (매 fold 마다 scaler 재fit) ─
    cv_scores = cross_val_score(pipeline, X_train_orig, y_train_orig, cv=min(5, len(X_train_orig) // len(np.unique(y_train_orig))))
    print(f"교차검증 정확도 (원본 train 기준, Pipeline): {cv_scores.mean():.2f} ± {cv_scores.std():.2f}\n")

    # ── 채널별 중요도 그래프 ──────────────────────────────────────
    print("📊 파장별 중요도 그래프 생성 중...")
    sc_temp  = StandardScaler()
    X_tr_sc  = sc_temp.fit_transform(X_train)
    temp_rf  = RandomForestClassifier(n_estimators=100, random_state=42)
    temp_rf.fit(X_tr_sc, y_train)

    importances = pd.Series(temp_rf.feature_importances_, index=CHANNELS)
    plt.figure(figsize=(10, 6))
    importances.sort_values().plot(kind='barh', color='skyblue')
    plt.title('채널별 중요도 (높을수록 플라스틱 분류에 중요한 파장)')
    plt.xlabel('중요도 (Importance)')
    plt.tight_layout()
    plt.savefig('importance_plot2.png')
    plt.show()

    # ── 모델 저장 (Pipeline 통째로 저장 — scaler + clf 일체) ───────
    with open('model_svm2.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    print("\n💾 모델 저장 완료: model_svm2.pkl  (scaler + clf 일체 저장)")
    print("   ℹ️  예측은 test2.py 를 사용하세요.\n")

    return pipeline


if __name__ == "__main__":
    train()

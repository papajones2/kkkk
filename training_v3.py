import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pickle

# ============================================================
# training_v3.py
#
# 원본 데이터(plastic_data_average.csv) 기준으로
#   1) 재질별 stratified 8:2 분리 (소수 클래스 누락 방지)
#   2) train 부분만 노이즈 증강
#   3) 증강된 train 으로 SVM 학습, 원본 test 로 평가
#
# PS(6개), PE(8개)처럼 샘플이 적은 클래스도
# stratify 옵션으로 train/test 양쪽에 비율대로 배정됨
# ============================================================

CHANNELS = [
    '410nm', '435nm', '460nm', '485nm', '510nm', '535nm',
    '560nm', '585nm', '610nm', '645nm', '680nm', '705nm',
    '730nm', '760nm', '810nm', '860nm', '900nm', '940nm',
    'slope_IR_VIS'
]

SPECTRAL_CHANNELS = CHANNELS[:-1]  # slope 제외한 18개 파장


def compute_slope(df):
    """slope_IR_VIS = (900nm + 940nm) / 2  /  (410nm + 435nm) / 2"""
    ir_mean  = (df['900nm'] + df['940nm']) / 2.0
    vis_mean = (df['410nm'] + df['435nm']) / 2.0
    return ir_mean / vis_mean.replace(0, 1e-10)


def load_data():
    fname = "plastic_data_average.csv"
    try:
        df = pd.read_csv(fname)
    except FileNotFoundError:
        print(f"❌ {fname} 없음. plastic.py 로 데이터를 먼저 수집하세요.")
        exit()

    # slope_IR_VIS 컬럼이 없으면 실시간 계산
    if 'slope_IR_VIS' not in df.columns:
        df['slope_IR_VIS'] = compute_slope(df)
        print("ℹ️  slope_IR_VIS 컬럼이 없어 자동 계산했습니다.")

    print(f"✅ 데이터 로드: {fname}  (총 {len(df)}개)")
    print(f"   재질별 원본 샘플 수:\n{df['label'].value_counts().to_string()}\n")

    if len(df['label'].unique()) < 2:
        print("❌ 최소 2종류 이상 플라스틱 데이터 필요")
        exit()

    return df


def stratified_split(X, y, test_size=0.2, random_state=42):
    """
    재질별로 비율을 유지하며 분리.
    소수 클래스(PS 6개, PE 8개)도 train/test 양쪽에 배정됨.
    """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(sss.split(X, y))
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx], train_idx, test_idx


def print_split_info(y_train_orig, y_test):
    print("📊 재질별 분리 결과 (증강 전 원본 기준):")
    classes = np.unique(np.concatenate([y_train_orig, y_test]))
    print(f"   {'재질':6s}  {'Train':>6s}  {'Test':>5s}  {'비율(Test%)':>10s}")
    print(f"   {'------':6s}  {'------':>6s}  {'-----':>5s}  {'----------':>10s}")
    for cls in classes:
        n_tr = (y_train_orig == cls).sum()
        n_te = (y_test      == cls).sum()
        total = n_tr + n_te
        print(f"   {cls:6s}  {n_tr:>6d}  {n_te:>5d}  {n_te/total:>9.0%}")
    print(f"   {'합계':6s}  {len(y_train_orig):>6d}  {len(y_test):>5d}")
    print()


def augment_train(X, y, n_copies=9, noise_std=0.02, random_state=42):
    """
    train 데이터만 증강 (원본 포함 n_copies+1 배).
    noise_std=0.02: 센서 실측 변동 범위 내 ±2% 노이즈.
    기본 n_copies=9 → 원본 포함 10배로 클래스 불균형 완화.
    """
    rng = np.random.RandomState(random_state)
    X_list = [X]
    y_list = [y]
    for _ in range(n_copies):
        noise = rng.normal(0, noise_std, X.shape)
        X_list.append(X + noise)
        y_list.append(y)
    return np.vstack(X_list), np.concatenate(y_list)


def plot_pca(X_train, y_train, X_test, y_test, filename='pca_plot3.png'):
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    sc  = StandardScaler()
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(sc.fit_transform(X_all))

    n_train = len(X_train)
    plt.figure(figsize=(9, 6))
    for lbl in np.unique(y_all):
        mask_tr = (y_all == lbl) & (np.arange(len(y_all)) <  n_train)
        mask_te = (y_all == lbl) & (np.arange(len(y_all)) >= n_train)
        plt.scatter(X_pca[mask_tr, 0], X_pca[mask_tr, 1],
                    label=f'{lbl} (train)', alpha=0.35, s=20)
        plt.scatter(X_pca[mask_te, 0], X_pca[mask_te, 1],
                    label=f'{lbl} (test)',  alpha=0.9,  s=60, marker='*')

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.legend(fontsize=8, ncol=2)
    plt.title('PCA 분포 (train=증강 포함 · test=원본만)\n덩어리가 잘 분리될수록 분류 정확도 높음')
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    print(f"   PCA 저장: {filename}  (PC1+PC2 설명 분산: {pca.explained_variance_ratio_.sum():.1%})\n")


def plot_importance(X_train, y_train, filename='importance_plot3.png'):
    sc     = StandardScaler()
    temp_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    temp_rf.fit(sc.fit_transform(X_train), y_train)

    importances = pd.Series(temp_rf.feature_importances_, index=CHANNELS)
    plt.figure(figsize=(10, 6))
    importances.sort_values().plot(kind='barh', color='skyblue')
    plt.title('채널별 중요도 (높을수록 플라스틱 분류에 중요한 파장)')
    plt.xlabel('중요도 (Importance)')
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    print(f"   중요도 저장: {filename}\n")


def train():
    df = load_data()

    X = df[CHANNELS].values
    y = df['label'].values

    # ── 1) 재질별 stratified 8:2 분리 ─────────────────────────────
    X_train_orig, X_test, y_train_orig, y_test, _, _ = stratified_split(
        X, y, test_size=0.2
    )
    print_split_info(y_train_orig, y_test)

    # ── 2) train 부분만 증강 (원본 포함 10배) ─────────────────────
    X_train, y_train = augment_train(X_train_orig, y_train_orig, n_copies=9)

    print(f"🔁 증강 완료: 원본 train {len(X_train_orig)}개 → 증강 후 {len(X_train)}개")
    aug_counts = pd.Series(y_train).value_counts().sort_index()
    print(f"   재질별 증강 후 train 수:\n{aug_counts.to_string()}\n")

    # ── 3) PCA 시각화 ──────────────────────────────────────────────
    print("📊 PCA 시각화 중...")
    plot_pca(X_train, y_train, X_test, y_test)

    # ── 4) SVM Pipeline 학습 ──────────────────────────────────────
    print("🤖 SVM 모델 학습 중...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    SVC(kernel='rbf', C=1.0, random_state=42, probability=True))
    ])
    pipeline.fit(X_train, y_train)

    # ── 5) 성능 평가 ───────────────────────────────────────────────
    y_pred_train = pipeline.predict(X_train)
    y_pred_test  = pipeline.predict(X_test)

    train_acc = (y_pred_train == y_train).mean() * 100
    test_acc  = (y_pred_test  == y_test ).mean() * 100

    print("\n" + "="*55)
    print("📊 정확도 (Accuracy)")
    print("="*55)
    print(f"🏋️  Train Accuracy : {train_acc:.2f}%  (증강 데이터 기준, 참고용)")
    print(f"🧪 Test Accuracy  : {test_acc:.2f}%  ← 실제 성능 지표 (원본만)")
    print("="*55)
    print("\n📋 상세 분류 성능 (Test set — 원본 데이터):")
    print(classification_report(y_test, y_pred_test, zero_division=0))

    # 교차검증은 원본 train 기준, Pipeline 사용 (매 fold 마다 scaler 재fit)
    min_class_count = pd.Series(y_train_orig).value_counts().min()
    cv_folds = min(5, min_class_count)
    if cv_folds >= 2:
        cv_scores = cross_val_score(pipeline, X_train_orig, y_train_orig, cv=cv_folds)
        print(f"교차검증 ({cv_folds}-fold, 원본 train 기준): "
              f"{cv_scores.mean():.2f} ± {cv_scores.std():.2f}\n")
    else:
        print("⚠️  교차검증 생략: 일부 클래스 샘플 수가 너무 적음\n")

    # ── 6) 채널 중요도 그래프 ─────────────────────────────────────
    print("📊 파장별 중요도 그래프 생성 중...")
    plot_importance(X_train, y_train)

    # ── 7) 모델 저장 ───────────────────────────────────────────────
    with open('model_svm3.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    print("💾 모델 저장 완료: model_svm3.pkl  (scaler + clf Pipeline)")
    print("   ℹ️  예측은 test2.py 에서 model_svm3.pkl 을 로드해서 사용하세요.\n")

    return pipeline


if __name__ == "__main__":
    train()

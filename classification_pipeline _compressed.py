import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

def load_dataset(path='P2_JPEG30_DN128mf3nj30vsmf5nj30U.npy'):
    data = np.load(path)
    X, y = data[:, :-1], data[:, -1]
    y = np.where(y == 1, 0, 1)  # Convert labels to 0/1 for AUC consistency
    return X, y

def evaluate_xgboost(X, y, n_splits=3, random_state=42):
    # Reduce number of splits to 3 to save time
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aucs, errs = [], []

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA to reduce dimensionality (retain 95% variance)
    pca = PCA(n_components=0.95, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    print(f"Reduced features from {X.shape[1]} to {X_pca.shape[1]} using PCA")

    # Feature selection on PCA-transformed data
    selector = SelectKBest(f_classif, k=min(20, X_pca.shape[1]))  # Select top 20 features or fewer

    for train_idx, test_idx in skf.split(X_pca, y):
        X_tr, X_te = X_pca[train_idx], X_pca[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Apply feature selection
        X_tr_selected = selector.fit_transform(X_tr, y_tr)
        X_te_selected = selector.transform(X_te)

        # XGBoost classifier with optimized parameters
        clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric='logloss',
            n_jobs=-1
        )

        # Fit without early_stopping_rounds for compatibility
        clf.fit(X_tr_selected, y_tr)

        # Use probability scores for AUC
        scores = clf.predict_proba(X_te_selected)[:, 1]
        preds = clf.predict(X_te_selected)

        auc = roc_auc_score(y_te, scores)
        err = np.mean(preds != y_te)
        
        aucs.append(auc)
        errs.append(err)

    return np.array(aucs), np.array(errs)

def main():
    X, y = load_dataset()
    aucs, errs = evaluate_xgboost(X, y)
    
    print(f"AUC (mean ± std): {aucs.mean():.4f} ± {aucs.std():.4f}")
    print(f"Error Rate: {errs.mean():.4f} ± {errs.std():.4f}")

if __name__ == '__main__':
    main()

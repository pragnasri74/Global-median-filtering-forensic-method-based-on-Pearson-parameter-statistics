# classification_pipeline.py

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def load_dataset(path='P2_DN128mf3nj30vsmf5nj30U.npy'):
    data = np.load(path)
    X, y = data[:, :-1], data[:, -1]
    y = np.where(y == 1, 0, 1)  # Convert labels to 0/1 for AUC consistency
    return X, y

def evaluate_svm_quadratic(X, y, n_splits=4, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aucs, errs = [], []

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        clf = SVC(kernel='poly', degree=2, random_state=random_state)
        clf.fit(X_tr, y_tr)

        scores = clf.decision_function(X_te)  # Faster than predict_proba
        preds = clf.predict(X_te)

        auc = roc_auc_score(y_te, scores)
        err = np.mean(preds != y_te)
        
        aucs.append(auc)
        errs.append(err)

    return np.array(aucs), np.array(errs)

def main():
    X, y = load_dataset()
    aucs, errs = evaluate_svm_quadratic(X, y)
    
    print(f"AUC (mean ± std): {aucs.mean():.4f} ± {aucs.std():.4f}")
    print(f"Error Rate: {errs.mean():.4f} ± {errs.std():.4f}")

if __name__ == '__main__':
    main()

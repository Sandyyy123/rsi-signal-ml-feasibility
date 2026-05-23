"""
Purged walk-forward cross-validation with embargo.
Prevents temporal leakage between train and test folds.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, precision_score
from sklearn.preprocessing import StandardScaler


def purged_train_mask(train_idx, test_idx, dates, embargo_days: int):
    """Remove samples from train that fall within embargo_days of test period."""
    test_start = dates[test_idx].min()
    embargo_cutoff = test_start - pd.Timedelta(days=embargo_days)
    return dates[train_idx] <= embargo_cutoff


def run_walkforward(X: pd.DataFrame, y: pd.Series, dates: pd.Index,
                    n_folds: int = 10, embargo_days: int = 5) -> dict:
    fold_size = len(X) // (n_folds + 1)
    aucs, precisions, fold_records = [], [], []

    for fold in range(n_folds):
        test_start = fold_size * (fold + 1)
        test_end = test_start + fold_size
        if test_end > len(X):
            break

        test_idx = np.arange(test_start, test_end)
        train_idx = np.arange(0, test_start)

        # Apply purge + embargo
        keep = purged_train_mask(train_idx, test_idx, dates, embargo_days)
        train_idx = train_idx[keep]

        if len(train_idx) < 50 or y.iloc[train_idx].sum() < 5:
            print(f"  Fold {fold+1}: skipped (insufficient positive labels in train)")
            continue

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42
        )
        model.fit(X_train_s, y_train)
        proba = model.predict_proba(X_test_s)[:, 1]

        if len(np.unique(y_test)) < 2:
            continue

        auc = roc_auc_score(y_test, proba)

        # Precision at top decile
        threshold = np.percentile(proba, 90)
        top_pred = (proba >= threshold).astype(int)
        prec = precision_score(y_test, top_pred, zero_division=0)

        aucs.append(auc)
        precisions.append(prec)
        fold_records.append({
            "fold": fold + 1,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "auc": round(auc, 4),
            "precision_top10": round(prec, 4),
            "pos_rate_test": round(float(y_test.mean()), 4),
        })
        print(f"  Fold {fold+1}: AUC={auc:.3f} | Prec@top10={prec:.3f} | pos_rate={y_test.mean():.2%}")

    mean_auc = float(np.mean(aucs)) if aucs else 0.5
    mean_prec = float(np.mean(precisions)) if precisions else 0.0

    # Simple walk-forward Sharpe proxy (signal return / vol)
    sharpe_proxy = (mean_auc - 0.5) / (np.std(aucs) + 1e-9) if len(aucs) > 1 else 0.0

    return {
        "mean_auc": mean_auc,
        "mean_precision_top10": mean_prec,
        "sharpe_proxy": float(sharpe_proxy),
        "fold_records": fold_records,
        "n_folds_completed": len(fold_records),
    }

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

RANDOM_STATE = 42

def univariate_cox_pvalue(x, time, event):
    df = pd.DataFrame({"x": x, "time": time, "event": event})
    cph = CoxPHFitter(penalizer=1e-4)
    cph.fit(df, duration_col="time", event_col="event")
    return cph.summary.loc["x", "p"]

def reduce_by_spearman_clusters(X, time, event, corr_cut=0.8):
    if X.shape[1] <= 1:
        return X.columns.tolist()

    corr = X.corr(method="spearman").abs().fillna(0)
    np.fill_diagonal(corr.values, 1.0)
    dist = 1.0 - corr
    Z = linkage(squareform(dist.values, checks=False), method="average")
    cluster_ids = fcluster(Z, t=1 - corr_cut, criterion="distance")

    kept = []
    for cid in np.unique(cluster_ids):
        cols = X.columns[cluster_ids == cid]
        best_col = min(cols, key=lambda c: univariate_cox_pvalue(X[c], time, event))
        kept.append(best_col)
    return kept

def lasso_cox_select(X, time, event, penalizers=None, n_splits=5):
    if penalizers is None:
        penalizers = np.logspace(-3, 0, 12)

    inner_cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    best_pen, best_score = None, -np.inf

    for pen in penalizers:
        scores = []
        for tr_idx, va_idx in inner_cv.split(X):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr_time, y_tr_event = time.iloc[tr_idx], event.iloc[tr_idx]
            y_va_time, y_va_event = time.iloc[va_idx], event.iloc[va_idx]

            scaler = StandardScaler()
            X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), columns=X.columns, index=X_tr.index)
            X_va_s = pd.DataFrame(scaler.transform(X_va), columns=X.columns, index=X_va.index)

            train_df = X_tr_s.copy()
            train_df["time"] = y_tr_time.values
            train_df["event"] = y_tr_event.values

            cph = CoxPHFitter(penalizer=pen, l1_ratio=1.0)
            cph.fit(train_df, duration_col="time", event_col="event")

            risk = cph.predict_partial_hazard(X_va_s).values.ravel()
            score = concordance_index(y_va_time, -risk, y_va_event)
            scores.append(score)

        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_pen = pen

    scaler = StandardScaler()
    X_s = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    train_df = X_s.copy()
    train_df["time"] = time.values
    train_df["event"] = event.values

    final_cph = CoxPHFitter(penalizer=best_pen, l1_ratio=1.0)
    final_cph.fit(train_df, duration_col="time", event_col="event")

    coefs = final_cph.params_
    selected = coefs[coefs.abs() > 1e-8].index.tolist()
    return selected, scaler, final_cph, best_pen, best_score

def cross_validated_feature_selection(df, feature_cols, time_col="time", event_col="event",
                                      outer_splits=5, var_threshold=0.1, corr_cut=0.8):
    X = df[feature_cols].copy()
    time = df[time_col].copy()
    event = df[event_col].copy()

    outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_risk = pd.Series(index=df.index, dtype=float)
    fold_cindex = []
    fold_features = []

    for fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X), start=1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr_time, y_te_time = time.iloc[tr_idx], time.iloc[te_idx]
        y_tr_event, y_te_event = event.iloc[tr_idx], event.iloc[te_idx]

        vt = VarianceThreshold(threshold=var_threshold)
        X_tr_v = pd.DataFrame(vt.fit_transform(X_tr), columns=X_tr.columns[vt.get_support()], index=X_tr.index)
        X_te_v = X_te[X_tr_v.columns]

        kept_cols = reduce_by_spearman_clusters(X_tr_v, y_tr_time, y_tr_event, corr_cut=corr_cut)
        X_tr_r = X_tr_v[kept_cols]
        X_te_r = X_te_v[kept_cols]

        selected, scaler, model, best_pen, inner_cindex = lasso_cox_select(X_tr_r, y_tr_time, y_tr_event)
        fold_features.append(selected)

        X_te_s = pd.DataFrame(scaler.transform(X_te_r[selected]), columns=selected, index=X_te_r.index)
        risk = model.predict_partial_hazard(X_te_s).values.ravel()
        oof_risk.loc[X_te.index] = risk

        cidx = concordance_index(y_te_time, -risk, y_te_event)
        fold_cindex.append(cidx)
        print(f"Fold {fold}: C-index={cidx:.3f}, selected_features={len(selected)}, best_pen={best_pen:.4f}")

    # Refit on the full ADNI derivation cohort
    vt = VarianceThreshold(threshold=var_threshold)
    X_v = pd.DataFrame(vt.fit_transform(X), columns=X.columns[vt.get_support()], index=X.index)

    kept_cols = reduce_by_spearman_clusters(X_v, time, event, corr_cut=corr_cut)
    X_r = X_v[kept_cols]

    final_selected, final_scaler, final_model, final_pen, final_inner_cindex = lasso_cox_select(X_r, time, event)

    print(f"\nOuter CV mean C-index: {np.mean(fold_cindex):.3f}")
    print(f"Final selected feature count: {len(final_selected)}")
    return {
        "oof_risk": oof_risk,
        "fold_cindex": fold_cindex,
        "fold_features": fold_features,
        "variance_kept_cols": X_v.columns.tolist(),
        "redundancy_kept_cols": kept_cols,
        "final_selected_features": final_selected,
        "final_scaler": final_scaler,
        "final_model": final_model,
        "final_penalizer": final_pen
    }

# Example
# radiomic_r2sn_cols = [col for col in adni_df.columns if col.startswith("feat_")]
# fs_result = cross_validated_feature_selection(adni_df, radiomic_r2sn_cols, time_col="time", event_col="event")
# arn_sig_features = fs_result["final_selected_features"]

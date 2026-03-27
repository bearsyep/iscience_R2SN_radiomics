import numpy as np
import pandas as pd
from sklearn.cluster import SpectralCoclustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from lifelines.statistics import multivariate_logrank_test

RANDOM_STATE = 42

def apply_admin_censor(df, time_col, event_col, max_time):
    out = df.copy()
    out[event_col] = np.where(out[time_col] > max_time, 0, out[event_col])
    out[time_col] = np.minimum(out[time_col], max_time)
    return out

def block_r2(X, row_labels, col_labels):
    X = np.asarray(X, dtype=float)
    X_hat = np.zeros_like(X)

    for r in np.unique(row_labels):
        r_mask = row_labels == r
        for c in np.unique(col_labels):
            c_mask = col_labels == c
            block_mean = X[np.ix_(r_mask, c_mask)].mean()
            X_hat[np.ix_(r_mask, c_mask)] = block_mean

    sse = np.sum((X - X_hat) ** 2)
    sst = np.sum((X - X.mean()) ** 2)
    return 1 - sse / sst if sst > 0 else 0.0

def minmax_scale(x):
    x = np.asarray(x, dtype=float)
    return np.zeros_like(x) if x.max() == x.min() else (x - x.min()) / (x.max() - x.min())

def bicluster_train_and_validate(train_df, valid_df, feature_cols,
                                 time_col="time", event_col="event",
                                 train_max_time=2160, valid_max_time=1460,
                                 k_range=range(3, 7)):
    train_df = apply_admin_censor(train_df, time_col, event_col, train_max_time)
    valid_df = apply_admin_censor(valid_df, time_col, event_col, valid_max_time)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_valid = scaler.transform(valid_df[feature_cols])

    records = []
    models = {}

    for k in k_range:
        model = SpectralCoclustering(n_clusters=k, random_state=RANDOM_STATE)
        model.fit(X_train)

        row_labels = model.row_labels_
        col_labels = model.column_labels_

        sil = silhouette_score(X_train, row_labels)
        logrank_stat = multivariate_logrank_test(
            train_df[time_col], row_labels, train_df[event_col]
        ).test_statistic
        br2 = block_r2(X_train, row_labels, col_labels)

        records.append({
            "k": k,
            "silhouette": sil,
            "logrank_stat": logrank_stat,
            "block_r2": br2
        })
        models[k] = model

    score_df = pd.DataFrame(records)
    score_df["silhouette_n"] = minmax_scale(score_df["silhouette"])
    score_df["logrank_n"] = minmax_scale(score_df["logrank_stat"])
    score_df["block_r2_n"] = minmax_scale(score_df["block_r2"])
    score_df["composite_score"] = score_df[["silhouette_n", "logrank_n", "block_r2_n"]].mean(axis=1)

    best_k = int(score_df.loc[score_df["composite_score"].idxmax(), "k"])
    best_model = models[best_k]
    train_labels = best_model.row_labels_

    centroids = np.vstack([
        X_train[train_labels == k].mean(axis=0)
        for k in range(best_k)
    ])
    valid_labels = cdist(X_valid, centroids, metric="euclidean").argmin(axis=1)

    train_logrank = multivariate_logrank_test(train_df[time_col], train_labels, train_df[event_col])
    valid_logrank = multivariate_logrank_test(valid_df[time_col], valid_labels, valid_df[event_col])

    print(score_df[["k", "silhouette", "logrank_stat", "block_r2", "composite_score"]])
    print("\nBest k:", best_k)
    print("Train overall log-rank p:", train_logrank.p_value)
    print("Valid overall log-rank p:", valid_logrank.p_value)

    return {
        "score_table": score_df,
        "best_k": best_k,
        "scaler": scaler,
        "model": best_model,
        "train_labels": train_labels,
        "valid_labels": valid_labels
    }

# Example
# arn_sig_features = fs_result["final_selected_features"]
# bicluster_result = bicluster_train_and_validate(
#     adni_df, nacc_df,
#     feature_cols=arn_sig_features,
#     time_col="time",
#     event_col="event"
# )

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
from sksurv.util import Surv
from sksurv.metrics import cumulative_dynamic_auc

def apply_admin_censor(df, time_col, event_col, max_time):
    out = df.copy()
    out[event_col] = np.where(out[time_col] > max_time, 0, out[event_col])
    out[time_col] = np.minimum(out[time_col], max_time)
    return out

def to_sksurv(df, time_col="time", event_col="event"):
    return Surv.from_arrays(event=df[event_col].astype(bool), time=df[time_col].astype(float))

def fit_and_validate_multivariable_cox(train_df, test_df, imaging_cols, clinical_cols,
                                       time_col="time", event_col="event",
                                       train_max_time=2190, test_max_time=1460,
                                       auc_times=(365, 730, 1095)):
    train_df = apply_admin_censor(train_df, time_col, event_col, train_max_time)
    test_df = apply_admin_censor(test_df, time_col, event_col, test_max_time)

    model_cols = imaging_cols + clinical_cols
    cph = CoxPHFitter()
    cph.fit(train_df[model_cols + [time_col, event_col]], duration_col=time_col, event_col=event_col)

    train_risk = cph.predict_partial_hazard(train_df[model_cols]).values.ravel()
    test_risk = cph.predict_partial_hazard(test_df[model_cols]).values.ravel()

    train_cindex = concordance_index(train_df[time_col], -train_risk, train_df[event_col])
    test_cindex = concordance_index(test_df[time_col], -test_risk, test_df[event_col])

    y_train = to_sksurv(train_df, time_col, event_col)
    y_test = to_sksurv(test_df, time_col, event_col)
    auc_times = np.array(auc_times, dtype=float)
    test_auc, test_mean_auc = cumulative_dynamic_auc(y_train, y_test, test_risk, auc_times)

    cutoff = np.median(train_risk)
    train_group = np.where(train_risk >= cutoff, "High", "Low")
    test_group = np.where(test_risk >= cutoff, "High", "Low")

    train_lr = logrank_test(
        train_df.loc[train_group == "High", time_col],
        train_df.loc[train_group == "Low", time_col],
        event_observed_A=train_df.loc[train_group == "High", event_col],
        event_observed_B=train_df.loc[train_group == "Low", event_col]
    )

    test_lr = logrank_test(
        test_df.loc[test_group == "High", time_col],
        test_df.loc[test_group == "Low", time_col],
        event_observed_A=test_df.loc[test_group == "High", event_col],
        event_observed_B=test_df.loc[test_group == "Low", event_col]
    )

    print("Train C-index:", round(train_cindex, 3))
    print("Test C-index :", round(test_cindex, 3))
    print("Time-dependent AUCs:")
    for t, auc in zip(auc_times.astype(int), test_auc):
        print(f"  {t} days: {auc:.3f}")
    print("Mean AUC:", round(test_mean_auc, 3))
    print("Train log-rank p:", train_lr.p_value)
    print("Test  log-rank p:", test_lr.p_value)

    # Optional PH assumption check
    # cph.check_assumptions(train_df[model_cols + [time_col, event_col]], p_value_threshold=0.05, show_plots=False)

    return {
        "model": cph,
        "train_risk": train_risk,
        "test_risk": test_risk,
        "cutoff": cutoff,
        "train_group": train_group,
        "test_group": test_group,
        "train_cindex": train_cindex,
        "test_cindex": test_cindex,
        "test_auc": dict(zip(auc_times.astype(int), test_auc)),
        "mean_auc": test_mean_auc,
        "summary": cph.summary
    }

# Example
# arn_sig_features = fs_result["final_selected_features"]
# clinical_cols = ["age", "sex", "education", "apoe4", "baseline_cog"]

# Model 1: ARN-Sig + clinical covariates
# model1_result = fit_and_validate_multivariable_cox(
#     adni_df, nacc_df,
#     imaging_cols=arn_sig_features,
#     clinical_cols=clinical_cols,
#     time_col="time",
#     event_col="event"
# )

# Model 2: volumetric features + clinical covariates
# volume_cols = [c for c in adni_df.columns if c.startswith("gmv_")]
# model2_result = fit_and_validate_multivariable_cox(adni_df, nacc_df, volume_cols, clinical_cols)

# Model 3: clinical-only model
# model3_result = fit_and_validate_multivariable_cox(adni_df, nacc_df, imaging_cols=[], clinical_cols=clinical_cols)

import pickle
import importlib.util
import numpy as np
import pandas as pd, sys, os

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    f1_score, average_precision_score, roc_auc_score,
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
import time

CONFIG_PATH = os.path.join(config.PREPROCESSING_DIR, "class_imbalance_config.py")
spec = importlib.util.spec_from_file_location("cfg", CONFIG_PATH)
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)

LEAKY_FEATURES = ["goes_flux", "goes_ordinal", "log_goes_flux"]
FEATURE_COLS   = [c for c in cfg.FEATURE_COLS if c not in LEAKY_FEATURES]

print("=" * 60)
print("  Solar Flare Forecasting — Step 4: Model Training")
print("=" * 60)
print(f"  Features       : {len(FEATURE_COLS)}")
print(f"  Leaky removed  : {LEAKY_FEATURES}")
print(f"  class_weight   : balanced  (≈ scale_pos_weight {cfg.SCALE_POS_WEIGHT})")
print("=" * 60)


print("\nLoading splits ...")
train = pd.read_csv(config.get_data_path("split_train.csv"), parse_dates=["timestamp"])
val   = pd.read_csv(config.get_data_path("split_val.csv"),   parse_dates=["timestamp"])
test  = pd.read_csv(config.get_data_path("split_test.csv"),  parse_dates=["timestamp"])

X_train = train[FEATURE_COLS].values
y_train = train["label_max"].values

X_val   = val[FEATURE_COLS].values
y_val   = val["label_max"].values

X_test  = test[FEATURE_COLS].values
y_test  = test["label_max"].values

print(f"  Train : {X_train.shape}  pos={y_train.sum():,} ({y_train.mean()*100:.1f}%)")
print(f"  Val   : {X_val.shape}  pos={y_val.sum():,} ({y_val.mean()*100:.1f}%)")
print(f"  Test  : {X_test.shape}  pos={y_test.sum():,} ({y_test.mean()*100:.1f}%)")


model = HistGradientBoostingClassifier(
    max_iter            = 1000,
    max_depth           = 6,
    learning_rate       = 0.05,
    min_samples_leaf    = 30,
    class_weight        = "balanced",
    early_stopping      = True,
    validation_fraction = 0.1,
    n_iter_no_change    = 30,
    random_state        = 42,
    verbose             = 1,
)


print("\nTraining model ...")
t0 = time.time()
model.fit(X_train, y_train)
train_time = time.time() - t0

n_trees = model.n_iter_
print(f"  Trees built    : {n_trees}  (stopped early)")
print(f"  Training time  : {train_time:.1f}s")


def evaluate(model, X, y, split_name, threshold=0.5):
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  
    tss         = sensitivity + specificity - 1

    expected = ((tp+fn)*(tp+fp) + (tn+fp)*(tn+fn)) / len(y)
    correct  = tp + tn
    hss      = (correct - expected) / (len(y) - expected) if (len(y) - expected) > 0 else 0

    return {
        "split"      : split_name,
        "threshold"  : threshold,
        "F1"         : f1_score(y, preds),
        "AUC-PR"     : average_precision_score(y, proba),
        "AUC-ROC"    : roc_auc_score(y, proba),
        "TSS"        : tss,
        "HSS"        : hss,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "TP"         : int(tp),
        "FP"         : int(fp),
        "TN"         : int(tn),
        "FN"         : int(fn),
        "proba"      : proba,
        "preds"      : preds,
    }


print("\nFinding optimal decision threshold on val set ...")
val_proba = model.predict_proba(X_val)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_val, val_proba)


f1_scores   = 2 * precisions[:-1] * recalls[:-1] / (
               precisions[:-1] + recalls[:-1] + 1e-9)
best_idx    = np.argmax(f1_scores)
best_thresh = thresholds[best_idx]
best_f1     = f1_scores[best_idx]

print(f"  Best threshold : {best_thresh:.4f}  (F1={best_f1:.4f})")
print(f"  Default (0.5)  : F1={f1_score(y_val, (val_proba>=0.5).astype(int)):.4f}")


print("\nEvaluating ...")
results_val  = evaluate(model, X_val,   y_val,   "Validation", best_thresh)
results_test = evaluate(model, X_test,  y_test,  "Test",       best_thresh)
results_val_default  = evaluate(model, X_val,  y_val,  "Val  (thresh=0.5)", 0.5)
results_test_default = evaluate(model, X_test, y_test, "Test (thresh=0.5)", 0.5)


print("\nComputing feature importance ...")

if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
else:
    
    perm = permutation_importance(
        model, X_val, y_val,
        n_repeats=5, random_state=42,
        scoring="average_precision"
    )
    importances = perm.importances_mean

feat_imp_df = pd.DataFrame({
    "feature"   : FEATURE_COLS,
    "importance": importances,
}).sort_values("importance", ascending=False).reset_index(drop=True)
feat_imp_df["rank"] = feat_imp_df.index + 1

print("\nTop 20 most important features:")
print(feat_imp_df.head(20).to_string(index=False))


def print_result(r):
    print(f"\n  ── {r['split']} (threshold={r['threshold']:.4f}) ──")
    print(f"  F1          : {r['F1']:.4f}")
    print(f"  AUC-PR      : {r['AUC-PR']:.4f}")
    print(f"  AUC-ROC     : {r['AUC-ROC']:.4f}")
    print(f"  TSS         : {r['TSS']:.4f}   (>0.5 = skilful, >0.9 = excellent)")
    print(f"  HSS         : {r['HSS']:.4f}")
    print(f"  Sensitivity : {r['sensitivity']:.4f}  (flare hit rate)")
    print(f"  Specificity : {r['specificity']:.4f}  (quiet-sun correct rejection)")
    print(f"  Confusion   : TP={r['TP']:,}  FP={r['FP']:,}  "
          f"TN={r['TN']:,}  FN={r['FN']:,}")
    miss_rate = r['FN'] / (r['TP'] + r['FN']) * 100
    fa_rate   = r['FP'] / (r['FP'] + r['TN']) * 100
    print(f"  Miss rate   : {miss_rate:.1f}%   (FN / all actual positives)")
    print(f"  False alarm : {fa_rate:.1f}%   (FP / all actual negatives)")

print("\n" + "=" * 60)
print("  RESULTS")
print("=" * 60)
print(f"\n  Model      : HistGradientBoostingClassifier")
print(f"  Trees      : {n_trees}")
print(f"  Train time : {train_time:.1f}s")
print_result(results_val)
print_result(results_test)
print_result(results_val_default)
print_result(results_test_default)


print("\nSaving outputs ...")

model_payload = {
    "model"          : model,
    "feature_cols"   : FEATURE_COLS,
    "threshold"      : best_thresh,
    "train_time_sec" : train_time,
    "n_trees"        : n_trees,
    "val_metrics"    : {k: v for k, v in results_val.items()
                        if k not in ("proba", "preds")},
    "test_metrics"   : {k: v for k, v in results_test.items()
                        if k not in ("proba", "preds")},
}
with open(config.get_model_path("solar_flare_model.pkl"), "wb") as f:
    pickle.dump(model_payload, f)
print("  solar_flare_model.pkl")

feat_imp_df.to_csv(config.get_data_path("feature_importance.csv"), index=False)
print("  feature_importance.csv")

# Save text report
report_lines = []
report_lines.append("Solar Flare Forecasting — Step 4 Results")
report_lines.append("=" * 60)
report_lines.append(f"Algorithm    : HistGradientBoostingClassifier")
report_lines.append(f"Trees built  : {n_trees}")
report_lines.append(f"Train time   : {train_time:.1f}s")
report_lines.append(f"Features     : {len(FEATURE_COLS)}")
report_lines.append(f"Threshold    : {best_thresh:.4f} (optimised on val F1)")
report_lines.append("")
for r in [results_val, results_test]:
    report_lines.append(f"── {r['split']} ──")
    for k in ["F1","AUC-PR","AUC-ROC","TSS","HSS",
               "sensitivity","specificity","TP","FP","TN","FN"]:
        report_lines.append(f"  {k:<14}: {r[k]}")
    report_lines.append("")
report_lines.append("Top 20 Features:")
report_lines.append(feat_imp_df.head(20).to_string(index=False))

with open(config.get_report_path("step4_results.txt"), "w") as f:
    f.write("\n".join(report_lines))
print("  step4_results.txt")

print("\n" + "=" * 60)
print("  Step 4 complete. Model saved to solar_flare_model.pkl")
print("  Load in future steps with:")
print("    import pickle")
print("    payload = pickle.load(open('solar_flare_model.pkl','rb'))")
print("    model   = payload['model']")
print("    thresh  = payload['threshold']")
print("=" * 60)

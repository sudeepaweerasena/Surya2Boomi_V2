import pickle
import config
from forecasting.live_data import get_live_snapshot

from forecasting.step7_72h_forecast import run as run_step7
from forecasting.step8_blackout_forecast import run as run_step8
from forecasting.step10_7day_forecast import run as run_step10
from forecasting.step11_7day_blackout import run as run_step11

MODEL_FILE     = config.get_model_path("solar_flare_model_multiclass.pkl")
LEAKY_FEATURES = ["goes_flux", "goes_ordinal", "log_goes_flux"]


def run_full_pipeline(verbose=True):

    # 1. Load model
    with open(MODEL_FILE, "rb") as f:
        payload = pickle.load(f)
    model     = payload["model"]
    feat_cols = [c for c in payload["feature_cols"] if c not in LEAKY_FEATURES]

    if verbose:
        print(f"Model: {payload['n_trees']} trees | {len(feat_cols)} features")

    # 2. Fetch live snapshot
    x0, now, source = get_live_snapshot(feat_cols, verbose=verbose)

    if verbose:
        print(f"Snapshot: {now} UTC  [source={source}]")

    # 3. Step 7: 72h flare forecast
    if verbose:
        print("Running 72-hour rollout (step7) ...")
    fh, fd, now, source = run_step7(model, feat_cols, x0, now, source, verbose)

    # 4. Step 8: 72h blackout
    if verbose:
        print("Computing 72h blackout probabilities (step8) ...")
    bh, bd = run_step8(fh)

    # 5. Step 10: 7-day flare forecast
    if verbose:
        print("Running 168-hour rollout (step10) ...")
    f7, _, _ = run_step10(model, feat_cols, x0, now, source, verbose)

    # 6. Step 11: 7-day blackout
    if verbose:
        print("Computing 7-day blackout probabilities (step11) ...")
    b7 = run_step11(f7)

    if verbose:
        print("Pipeline complete — all data in memory.")

    return {
        "fh": fh,
        "fd": fd,
        "bh": bh,
        "bd": bd,
        "f7": f7,
        "b7": b7,
        "source": source,
    }

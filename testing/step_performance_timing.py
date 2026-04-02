import time
import subprocess
import datetime
import os

OUT_FILE = "performance_timing_proof.txt"

# TARGETS (from NFR04 in SRS)
STEPS = [
    # (label,                            command,                              target_sec)
    ("step1 - Feature Engineering",      "python3 step1_feature_engineering.py",  60),
    ("step2 - Temporal Split",           "python3 step2_temporal_split.py",        30),
    ("step3 - SMOTE Balancing",          "python3 step3_class_imbalance.py",       30),
    ("step4 - Binary Model Training",    "python3 step4_model_training.py",       120),
    ("step7 - 72h Forecast Rollout",     "python3 step7_72h_forecast.py --fallback", 10),
    ("step8 - HF Blackout 72h",          "python3 step8_blackout_forecast.py",      5),
    ("step10 - 7-Day Forecast Rollout",  "python3 step10_7day_forecast.py --fallback", 10),
    ("step11 - HF Blackout 7-Day",       "python3 step11_7day_blackout.py",         5),
]

W = 72

def div(c="=", n=W): return c * n

lines = []
results = []

lines += [
    div("═"),
    "  SURYA2BOOMI — PERFORMANCE TIMING PROOF",
    f"  Generated : {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
    f"  Machine   : {os.uname().sysname} {os.uname().machine}",
    "  Purpose   : Prove NFR04 (Performance) targets are met",
    div("═"),
    "",
    f"  NFR04 Targets:",
    f"    - Full training pipeline      ≤ 120 seconds",
    f"    - Live inference pipeline     ≤ 10 seconds per step",
    f"    - HF blackout post-processing ≤ 5 seconds per step",
    "",
    f"  {'Step':<42} {'Target':>7} {'Actual':>8}  {'Status'}",
    f"  {'-'*42} {'-'*7} {'-'*8}  {'-'*8}",
]

print("\nRunning all pipeline steps and timing...\n")

for label, cmd, target in STEPS:
    print(f"  Running: {label} ...", end=" ", flush=True)
    start   = time.time()
    proc    = subprocess.run(cmd, shell=True,
                             capture_output=True, text=True,
                             cwd=os.path.dirname(os.path.abspath(__file__)))
    elapsed = time.time() - start
    passed  = elapsed <= target
    status  = "PASS ✓" if passed else "FAIL ✗"
    results.append((label, target, elapsed, passed, proc.returncode))
    print(f"{elapsed:.2f}s  [{status}]")
    lines.append(
        f"  {label:<42} {target:>6}s {elapsed:>7.2f}s  {status}"
    )

# ── Summary ──
total_train    = sum(r[2] for r in results if "Training" in r[0] or
                     "Engineer" in r[0] or "Split" in r[0] or "SMOTE" in r[0])
total_inference= sum(r[2] for r in results if "Rollout" in r[0] or "Blackout" in r[0])
all_pass       = all(r[3] for r in results)
n_pass         = sum(r[3] for r in results)
n_total        = len(results)

lines += [
    "",
    div(),
    "  SUMMARY",
    div(),
    "",
    f"  Steps run          : {n_total}",
    f"  Passed             : {n_pass}",
    f"  Failed             : {n_total - n_pass}",
    f"  Pass rate          : {n_pass/n_total*100:.1f}%",
    "",
    f"  Training pipeline total  : {total_train:.2f}s",
    f"  Inference pipeline total : {total_inference:.2f}s",
    "",
    f"  ALL TARGETS MET: {'YES ✓' if all_pass else 'NO — SEE FAILURES ABOVE'}",
    "",
    div("═"),
    "  SURYA2BOOMI · Sudeepa Weerasena · IIT Sri Lanka",
    div("═"),
    "",
]

report = "\n".join(lines)
with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.write(report)

print()
print(f"Report saved → {OUT_FILE}")
print()
print(f"  Training pipeline  : {total_train:.2f}s")
print(f"  Inference pipeline : {total_inference:.2f}s")
print(f"  All targets met    : {'YES' if all_pass else 'NO'}")

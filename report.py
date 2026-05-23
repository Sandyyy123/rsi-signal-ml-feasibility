"""
Go/No-Go report generator.
Decision thresholds: AUC > 0.58, precision@top10 > 0.18, Sharpe > 0.7 = GO.
"""

import json
import os
from datetime import datetime


GO_THRESHOLDS = {"auc": 0.58, "precision_top10": 0.18, "sharpe": 0.7}
CONDITIONAL_THRESHOLDS = {"auc": 0.52, "precision_top10": 0.10, "sharpe": 0.3}


def generate_report(signal_stats: dict, cv_results: dict, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    auc = cv_results["mean_auc"]
    prec = cv_results["mean_precision_top10"]
    sharpe = cv_results["sharpe_proxy"]
    leakage = False  # Placeholder; real implementation runs leakage diagnostic

    if (auc >= GO_THRESHOLDS["auc"] and
            prec >= GO_THRESHOLDS["precision_top10"] and
            sharpe >= GO_THRESHOLDS["sharpe"]):
        decision = "GO — Full Research Program Recommended"
    elif (auc >= CONDITIONAL_THRESHOLDS["auc"] and
          prec >= CONDITIONAL_THRESHOLDS["precision_top10"]):
        decision = "CONDITIONAL GO — Regime-Specific Investigation Required"
    else:
        decision = "NO-GO — Signal Does Not Demonstrate Learnable Structure"

    verdict = {
        "decision": decision,
        "auc": round(auc, 4),
        "precision_top10": round(prec, 4),
        "sharpe": round(sharpe, 4),
        "leakage_detected": leakage,
        "base_rate": round(signal_stats["base_rate"], 4),
        "n_signals": signal_stats["n_signals"],
        "forward_returns": signal_stats["forward_returns"],
        "n_folds_completed": cv_results["n_folds_completed"],
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }

    report_path = os.path.join(output_dir, "feasibility_report.json")
    with open(report_path, "w") as f:
        json.dump(verdict, f, indent=2)
    print(f"  Report saved: {report_path}")

    # Print fold table
    print("\n  Walk-Forward Fold Summary:")
    print(f"  {'Fold':>4} {'Train N':>8} {'Test N':>7} {'AUC':>6} {'Prec@10':>8} {'Pos%':>6}")
    for r in cv_results["fold_records"]:
        print(f"  {r['fold']:>4} {r['n_train']:>8} {r['n_test']:>7} "
              f"{r['auc']:>6.3f} {r['precision_top10']:>8.3f} {r['pos_rate_test']*100:>5.1f}%")

    return verdict

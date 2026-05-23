"""
RSI Peak-to-Trough ML Feasibility Study
Entry point: runs signal characterization, feature engineering, and walk-forward CV.
Usage: python main.py --ticker SPY --start 2010-01-01 --end 2023-12-31
"""

import argparse
import sys
from signal_characterizer import characterize_signal
from feature_engineer import build_features
from walkforward_cv import run_walkforward
from report import generate_report


def main():
    parser = argparse.ArgumentParser(description="RSI P2T ML Feasibility Study")
    parser.add_argument("--ticker", default="SPY", help="Yahoo Finance ticker")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2023-12-31")
    parser.add_argument("--rsi-period", type=int, default=14)
    parser.add_argument("--trough-threshold", type=float, default=30.0)
    parser.add_argument("--peak-threshold", type=float, default=70.0)
    parser.add_argument("--embargo-days", type=int, default=5,
                        help="Embargo window between train/test to prevent leakage")
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--output-dir", default="./output")
    args = parser.parse_args()

    print(f"\n[1/4] Characterizing signal: {args.ticker} RSI-{args.rsi_period} "
          f"trough<{args.trough_threshold} -> peak>{args.peak_threshold}")
    signal_stats = characterize_signal(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        rsi_period=args.rsi_period,
        trough_threshold=args.trough_threshold,
        peak_threshold=args.peak_threshold,
    )

    print(f"\n[2/4] Building leakage-safe features (embargo={args.embargo_days}d)")
    X, y, dates = build_features(signal_stats["ohlcv"], signal_stats["labels"])

    print(f"\n[3/4] Running purged walk-forward CV ({args.n_folds} folds)")
    cv_results = run_walkforward(X, y, dates, n_folds=args.n_folds,
                                 embargo_days=args.embargo_days)

    print(f"\n[4/4] Generating go/no-go report -> {args.output_dir}/")
    verdict = generate_report(signal_stats, cv_results, args.output_dir)

    print(f"\n{'='*50}")
    print(f"VERDICT: {verdict['decision']}")
    print(f"  Out-of-sample AUC : {verdict['auc']:.3f}")
    print(f"  Precision@top-decile: {verdict['precision_top10']:.3f}")
    print(f"  Walk-forward Sharpe : {verdict['sharpe']:.2f}")
    print(f"  Leakage detected   : {verdict['leakage_detected']}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()

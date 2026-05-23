# RSI Peak-to-Trough ML Feasibility Study

A leakage-proof framework for evaluating whether RSI peak-to-trough reversal signals carry learnable structure for ML models.

## Architecture

```
main.py
├── signal_characterizer.py  — RSI computation, P2T label detection, base rate analysis
├── feature_engineer.py      — Lag-safe feature construction (all features shifted t-1)
├── walkforward_cv.py        — Purged walk-forward CV with embargo window
└── report.py                — Go/No-Go verdict against explicit AUC/Sharpe thresholds
```

## Key Design Decisions

**Leakage prevention**
- All features lagged by minimum 1 bar at construction time
- Embargo period between train and test folds (default: 5 trading days)
- Labels detected using only data available at prediction time

**Go/No-Go thresholds**
| Verdict | AUC | Precision@Top-Decile | Walk-Forward Sharpe |
|---|---|---|---|
| GO | > 0.58 | > 18% | > 0.7 |
| Conditional GO | 0.52-0.58 | > 10% | > 0.3 |
| No-Go | < 0.52 | - | - |

## Setup

```bash
pip install -r requirements.txt
python main.py --ticker SPY --start 2010-01-01 --end 2023-12-31 --rsi-period 14
```

Works without yfinance (demo mode with synthetic data) for CI/testing.

## Output

`output/feasibility_report.json` — machine-readable verdict with all metrics.

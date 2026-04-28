"""
compute_momentum_baseline.py — Stage 2 momentum baseline (v2.3.7)

Computes "proper momentum" baseline (leakage-free) on the same fold splits
as Stage 2 retrain. No NN training — just signal/realized split per ticker.

Method:
  For each fold:
    For each test ticker:
      Sort snapshots by snap_idx (chronological per ticker)
      Split in half: early = signal, late = realized
      momentum_signal[ticker] = mean(early Y_ret)
      momentum_realized[ticker] = mean(late Y_ret)
    Top 5 by momentum_signal -> compute alpha vs:
      - Universe equal-weight (mean of all test tickers' realized)
      - SPY 3-month forward return

Outputs:
  results/momentum_baseline_v237.json (both with/without SNDK configs)

Comparison: Stage 2 NN edge over momentum (per fold + aggregate).

Usage:
  python compute_momentum_baseline.py
"""

import json
import sys
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, '.')

CACHE_PATH = 'results/backtest_cache.npz'
OUTPUT_JSON = Path('results/momentum_baseline_v237.json')
N_SELECT = 5
N_FOLDS = 5

# Two configurations to match Stage 2 sensitivity analysis
CONFIGURATIONS = [
    {'name': 'without_sndk', 'excluded_tickers': {'SNDK'},
     'stage2_dir': Path('results/stage2/top1_trial58')},
    {'name': 'with_sndk', 'excluded_tickers': set(),
     'stage2_dir': Path('results/stage2_with_sndk/top1_trial58')},
]


def load_cache_filtered(excluded_tickers):
    """Load cache, optionally filter tickers."""
    print(f"[Data] Loading {CACHE_PATH}...")
    data = np.load(CACHE_PATH, allow_pickle=True)
    Y_ret = data['Y_ret']
    meta = data['meta']  # ndarray (N, 3): [ticker, snap_idx, date_str]

    sample_tickers = meta[:, 0].astype(str)
    snap_idx = meta[:, 1].astype(int)

    if excluded_tickers:
        mask = ~np.isin(sample_tickers, list(excluded_tickers))
        n_excluded = int((~mask).sum())
        Y_ret = Y_ret[mask]
        sample_tickers = sample_tickers[mask]
        snap_idx = snap_idx[mask]
        print(f"[Data] Excluded {n_excluded} samples from "
              f"{sorted(excluded_tickers)}")

    print(f"[Data] {len(Y_ret):,} samples, "
          f"{len(set(sample_tickers))} unique tickers")
    return Y_ret, sample_tickers, snap_idx


def get_folds():
    """Get the same stratified k-fold split used by Stage 2."""
    from backtest import _stratified_kfold, _get_ticker_sectors

    # We need the unique tickers in the *filtered* cache,
    # but folds are computed per-config (since SNDK presence affects ticker count).
    # Returns a function that builds folds from a ticker list.
    def build_folds(unique_tickers):
        ticker_sectors = _get_ticker_sectors(unique_tickers, verbose=False)
        return _stratified_kfold(unique_tickers, ticker_sectors, N_FOLDS)

    return build_folds


def compute_momentum_for_fold(test_ticker_list, sample_tickers, snap_idx, Y_ret):
    """For each test ticker, split snapshots into early/late, compute
    momentum signal and realized return.

    Returns:
        ticker_signal: {ticker -> mean(early Y_ret)}
        ticker_realized: {ticker -> mean(late Y_ret)}
    """
    test_set = set(test_ticker_list)

    # Group sample indices by ticker
    ticker_indices = defaultdict(list)
    for i, tk in enumerate(sample_tickers):
        if tk in test_set:
            ticker_indices[tk].append(i)

    ticker_signal = {}
    ticker_realized = {}

    for tk, idx_list in ticker_indices.items():
        if len(idx_list) < 4:  # need at least 2 early + 2 late
            continue
        # Sort by snap_idx (chronological per ticker)
        sorted_idx = sorted(idx_list, key=lambda i: snap_idx[i])
        n_half = len(sorted_idx) // 2
        early_idx = sorted_idx[:n_half]
        late_idx = sorted_idx[n_half:]

        ticker_signal[tk] = float(np.mean([Y_ret[i] for i in early_idx]))
        ticker_realized[tk] = float(np.mean([Y_ret[i] for i in late_idx]))

    return ticker_signal, ticker_realized


def compute_universe_mean(test_ticker_list, sample_tickers, Y_ret):
    """Universe mean realized return for the test set
    (matches Stage 2's all_mean_return)."""
    test_set = set(test_ticker_list)
    rets = [Y_ret[i] for i, tk in enumerate(sample_tickers) if tk in test_set]
    return float(np.mean(rets)) if rets else 0.0


def load_stage2_fold_results(stage2_dir):
    """Load Stage 2 per-fold results for direct comparison."""
    summary_path = stage2_dir / 'summary.json'
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        summary = json.load(f)
    return summary


def run_one_config(config):
    """Run momentum baseline for one configuration (with/without SNDK)."""
    name = config['name']
    excluded = config['excluded_tickers']
    stage2_dir = config['stage2_dir']

    print(f"\n{'=' * 60}")
    print(f"Configuration: {name}")
    print(f"{'=' * 60}")

    Y_ret, sample_tickers, snap_idx = load_cache_filtered(excluded)
    unique_tickers = sorted(set(sample_tickers))
    build_folds = get_folds()
    folds = build_folds(unique_tickers)

    stage2 = load_stage2_fold_results(stage2_dir)

    fold_results = []
    for fold_id in range(N_FOLDS):
        train_set, test_set = folds[fold_id]
        test_tickers = sorted(test_set)

        signal, realized = compute_momentum_for_fold(
            test_tickers, sample_tickers, snap_idx, Y_ret
        )

        # Top 5 by signal
        if len(signal) < N_SELECT:
            print(f"  Fold {fold_id+1}: only {len(signal)} tickers with "
                  f"sufficient snapshots, skipping")
            continue

        top5 = sorted(signal.keys(), key=lambda tk: -signal[tk])[:N_SELECT]
        top5_realized_mean = float(np.mean([realized[tk] for tk in top5]))

        # Universe mean (using late half for fairness — same realized half)
        all_realized = [realized[tk] for tk in realized]
        universe_late_mean = float(np.mean(all_realized))

        # Also compute universe mean using full Y_ret (matches Stage 2 alpha def)
        universe_full_mean = compute_universe_mean(
            test_tickers, sample_tickers, Y_ret
        )

        # Momentum alpha: vs late-half universe mean (apples to apples
        # for momentum baseline since that's what momentum operates on)
        momentum_alpha_late = top5_realized_mean - universe_late_mean

        # Momentum alpha vs Stage 2's alpha definition (full Y_ret universe)
        momentum_alpha_full = top5_realized_mean - universe_full_mean

        # Stage 2 NN alpha for this fold (for comparison)
        stage2_alpha = None
        stage2_top5 = None
        stage2_rank_corr = None
        if stage2 is not None:
            for fold_info in stage2['per_fold']:
                if fold_info['fold_id'] == fold_id + 1:
                    stage2_alpha = fold_info['selection_alpha']
                    stage2_top5 = fold_info['top5']
                    stage2_rank_corr = fold_info['rank_corr']
                    break

        nn_edge = (stage2_alpha - momentum_alpha_full) if stage2_alpha is not None else None

        result = {
            'fold_id': fold_id + 1,
            'n_test_tickers': len(test_tickers),
            'n_with_signal': len(signal),
            'momentum_top5': top5,
            'momentum_top5_realized_mean': top5_realized_mean,
            'universe_late_mean': universe_late_mean,
            'universe_full_mean': universe_full_mean,
            'momentum_alpha_vs_late_universe': momentum_alpha_late,
            'momentum_alpha_vs_full_universe': momentum_alpha_full,
            'stage2_nn_top5': stage2_top5,
            'stage2_nn_alpha': stage2_alpha,
            'stage2_nn_rank_corr': stage2_rank_corr,
            'nn_edge_vs_momentum_pp': nn_edge,
        }
        fold_results.append(result)

        print(f"\n  Fold {fold_id+1}/5  ({len(test_tickers)} test, "
              f"{len(signal)} with signal)")
        print(f"    Momentum top 5: {top5}")
        print(f"    Momentum α (vs full universe): "
              f"{momentum_alpha_full*100:+.1f}%p")
        if stage2_alpha is not None:
            print(f"    Stage 2 NN top 5: {stage2_top5}")
            print(f"    Stage 2 NN α:                 {stage2_alpha*100:+.1f}%p")
            print(f"    NN edge over momentum:        {nn_edge*100:+.1f}%p"
                  f"   {'✓' if nn_edge > 0 else '✗'}")

    # Aggregate
    if fold_results:
        nn_edges = [r['nn_edge_vs_momentum_pp'] for r in fold_results
                    if r['nn_edge_vs_momentum_pp'] is not None]
        momentum_alphas = [r['momentum_alpha_vs_full_universe']
                           for r in fold_results]
        nn_alphas = [r['stage2_nn_alpha'] for r in fold_results
                     if r['stage2_nn_alpha'] is not None]

        aggregate = {
            'mean_momentum_alpha': float(np.mean(momentum_alphas)),
            'std_momentum_alpha': float(np.std(momentum_alphas)),
            'mean_nn_alpha': float(np.mean(nn_alphas)) if nn_alphas else None,
            'mean_nn_edge_pp': float(np.mean(nn_edges)) if nn_edges else None,
            'std_nn_edge_pp': float(np.std(nn_edges)) if nn_edges else None,
            'nn_wins': int(sum(1 for e in nn_edges if e > 0)),
            'momentum_wins': int(sum(1 for e in nn_edges if e <= 0)),
            'n_folds_compared': len(nn_edges),
        }

        print(f"\n  --- Aggregate ({name}) ---")
        print(f"    Momentum α (mean):  "
              f"{aggregate['mean_momentum_alpha']*100:+.2f}%p ± "
              f"{aggregate['std_momentum_alpha']*100:.2f}%p")
        if aggregate['mean_nn_alpha'] is not None:
            print(f"    NN α (mean):        "
                  f"{aggregate['mean_nn_alpha']*100:+.2f}%p")
            print(f"    NN edge over momentum:  "
                  f"{aggregate['mean_nn_edge_pp']*100:+.2f}%p ± "
                  f"{aggregate['std_nn_edge_pp']*100:.2f}%p  "
                  f"({aggregate['nn_wins']}/{aggregate['n_folds_compared']} folds)")
    else:
        aggregate = None

    return {
        'config_name': name,
        'excluded_tickers': sorted(excluded),
        'per_fold': fold_results,
        'aggregate': aggregate,
    }


def main():
    print("=" * 70)
    print("MOMENTUM BASELINE — Stage 2 v2.3.7 (apples-to-apples)")
    print("=" * 70)

    all_results = {}
    for config in CONFIGURATIONS:
        result = run_one_config(config)
        all_results[config['name']] = result

    # Save
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n[Output] {OUTPUT_JSON}")
    print()
    print("=" * 70)
    print("SUMMARY (for v2.3.7 README)")
    print("=" * 70)
    for name, result in all_results.items():
        agg = result['aggregate']
        if agg is None:
            continue
        print(f"\n{name}:")
        print(f"  Stage 2 NN α:      {agg['mean_nn_alpha']*100:+.2f}%p")
        print(f"  Momentum α:        {agg['mean_momentum_alpha']*100:+.2f}%p")
        print(f"  NN edge:           "
              f"{agg['mean_nn_edge_pp']*100:+.2f}%p "
              f"({agg['nn_wins']}/{agg['n_folds_compared']} folds NN wins)")


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""v2.3.17 purged expanding walk-forward — deployment gate.

Implements pre_registration_v2317_walkforward.md sections 2-3.
Calendar-year folds, purge 63 trading days (92 calendar-day conservative
bound), production Trial 52 config read from config.py, N=20 ensemble.
Primary metrics computed per snapshot date: IC and top-5 alpha.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from scipy.stats import spearmanr

import config as config_module
from models import HeteroscedasticDualHeadNN, heteroscedastic_loss

SEED = 42
LOG_EPSILON = 1e-4
CACHE = 'results/backtest_cache.npz'
EXCLUDED_TICKERS = {'SNDK'}
RESULTS_ROOT = Path('results/walkforward_v2317')
LOSS_FN = heteroscedastic_loss

# (fold_id, test_year, binding). Fold 5 test includes the 2026-01 cache tail.
FOLD_SPEC = [
    (0, 2020, False),
    (1, 2021, True),
    (2, 2022, True),
    (3, 2023, True),
    (4, 2024, True),
    (5, 2025, True),
]

# 63 trading days <= 92 calendar days (63 weekdays = 88.2 cal + holidays);
# purging on the calendar bound is strictly conservative.
PURGE_CAL_DAYS = 92
MIN_DATE_COUNT = 30   # min tickers at a snapshot date for IC/alpha
N_SELECT = 5

MOMENTUM_CANDIDATES = ['ret_180d', 'return_180d', 'ret_1y', 'return_1y',
                       'ret_90d', 'return_90d', 'ret_60d']


def load_data(cache_path=CACHE):
    data = np.load(cache_path, allow_pickle=True)
    X = data['X'].astype(np.float32)
    Y_ret = data['Y_ret'].astype(np.float32)
    Y_risk = data['Y_risk'].astype(np.float32)
    meta = data['meta']
    feat_names = [str(f) for f in data['feat_names']]
    sample_tickers = meta[:, 0].astype(str)
    sample_dates = meta[:, 2].astype(str)

    if EXCLUDED_TICKERS:
        mask = ~np.isin(sample_tickers, list(EXCLUDED_TICKERS))
        n_total = len(X)
        X = X[mask]
        Y_ret = Y_ret[mask]
        Y_risk = Y_risk[mask]
        sample_tickers = sample_tickers[mask]
        sample_dates = sample_dates[mask]
        print(f"[Data] Excluded {n_total - len(X)} samples from "
              f"{sorted(EXCLUDED_TICKERS)} (was {n_total}, now {len(X)})")

    print(f"[Data] {len(X):,} samples x {X.shape[1]} features, "
          f"{len(set(sample_tickers))} tickers, "
          f"date range {min(sample_dates)} ~ {max(sample_dates)}")
    return {
        'X': X, 'Y_ret': Y_ret, 'Y_risk': Y_risk,
        'sample_tickers': np.array(sample_tickers),
        'sample_dates': np.array(sample_dates),
        'feat_names': feat_names,
    }


def make_fold_masks(sample_dates, fold_id, test_year, last_fold):
    test_start = f"{test_year}-01-01"
    if last_fold:
        test_mask = sample_dates >= test_start
    else:
        test_end = f"{test_year}-12-31"
        test_mask = (sample_dates >= test_start) & (sample_dates <= test_end)
    purge_cutoff = (pd.Timestamp(test_start)
                    - pd.Timedelta(days=PURGE_CAL_DAYS)).strftime('%Y-%m-%d')
    train_mask = sample_dates <= purge_cutoff
    return train_mask, test_mask, purge_cutoff


def pick_momentum_feature(feat_names):
    for cand in MOMENTUM_CANDIDATES:
        if cand in feat_names:
            return feat_names.index(cand), cand
    # fallback: any name containing 'ret' and '180' or '90'
    for horizon in ['180', '90', '60']:
        for i, f in enumerate(feat_names):
            fl = f.lower()
            if ('ret' in fl or 'return' in fl) and horizon in fl:
                return i, f
    return None, None


def per_date_metrics(dates, preds, actuals, mom_signal=None):
    """Group by exact snapshot date; IC and top-5 alpha per date."""
    rows = []
    order = np.argsort(dates)
    dates_s = dates[order]
    preds_s = preds[order]
    actuals_s = actuals[order]
    mom_s = mom_signal[order] if mom_signal is not None else None

    uniq, starts = np.unique(dates_s, return_index=True)
    bounds = list(starts) + [len(dates_s)]
    for i, d in enumerate(uniq):
        lo, hi = bounds[i], bounds[i + 1]
        n = hi - lo
        if n < MIN_DATE_COUNT:
            continue
        p = preds_s[lo:hi]
        a = actuals_s[lo:hi]
        ic = spearmanr(p, a)[0]
        if not np.isfinite(ic):
            continue
        top_idx = np.argsort(p)[::-1][:N_SELECT]
        alpha = float(a[top_idx].mean() - a.mean())
        row = {'date': str(d), 'n': int(n),
               'ic': float(ic), 'alpha': alpha,
               'top5_mean_ret': float(a[top_idx].mean()),
               'universe_mean_ret': float(a.mean())}
        if mom_s is not None:
            row['momentum_ic'] = float(np.nan_to_num(spearmanr(mom_s[lo:hi], a)[0]))
        rows.append(row)
    return rows


def fetch_spy_forward(dates_needed):
    """SPY 63-trading-day forward return keyed by snapshot date."""
    import yfinance as yf
    hist = yf.Ticker('SPY').history(period='12y', auto_adjust=True)
    if hist is None or len(hist) == 0:
        raise RuntimeError('SPY fetch empty')
    idx = hist.index.tz_localize(None).normalize()
    close = hist['Close'].values
    date_to_pos = {d.strftime('%Y-%m-%d'): i for i, d in enumerate(idx)}
    out = {}
    for d in dates_needed:
        pos = date_to_pos.get(d)
        if pos is None:
            # snapshot date may fall on a non-SPY day; take next available
            later = idx[idx >= pd.Timestamp(d)]
            if len(later) == 0:
                continue
            pos = date_to_pos[later[0].strftime('%Y-%m-%d')]
        if pos + 63 < len(close):
            out[d] = float(close[pos + 63] / close[pos] - 1)
    return out


def run_fold(data, fold_id, test_year, binding, last_fold,
             seed_override, n_ensemble, epochs_cap=None, out_dir=None):
    t0 = time.time()
    train_mask, test_mask, purge_cutoff = make_fold_masks(
        data['sample_dates'], fold_id, test_year, last_fold)

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise RuntimeError(f"Empty split for fold {fold_id}")

    X = data['X']
    X_tr_full = X[train_mask]
    Y_ret_tr_full = data['Y_ret'][train_mask]
    Y_risk_tr_full_log = np.log(np.maximum(data['Y_risk'][train_mask],
                                           LOG_EPSILON))
    X_te = X[test_mask]
    Y_ret_te = data['Y_ret'][test_mask]
    te_dates = data['sample_dates'][test_mask]

    print(f"    Train: {train_mask.sum():,} (<= {purge_cutoff}), "
          f"Test: {test_mask.sum():,} ({test_year}"
          f"{'+' if last_fold else ''})")

    # Feature selection on fold train only
    var_thr = getattr(config_module, 'VAR_THRESHOLD', 0.01)
    corr_thr = getattr(config_module, 'CORR_THRESHOLD', 0.05)
    var_per_feat = X_tr_full.var(axis=0)
    keep_var = var_per_feat > var_thr
    corr_per_feat = np.array([
        abs(np.corrcoef(X_tr_full[:, j], Y_ret_tr_full)[0, 1])
        if X_tr_full[:, j].std() > 0 else 0
        for j in range(X_tr_full.shape[1])
    ])
    corr_per_feat = np.nan_to_num(corr_per_feat, nan=0)
    keep = keep_var & (corr_per_feat > corr_thr)
    if keep.sum() < 10:
        keep = keep_var
    if keep.sum() < 10:
        keep = np.ones(X.shape[1], dtype=bool)

    mom_idx, mom_name = pick_momentum_feature(data['feat_names'])
    mom_signal = X_te[:, mom_idx].copy() if mom_idx is not None else None

    X_tr_sel = X_tr_full[:, keep]
    X_te_sel = X_te[:, keep]
    n_features = int(keep.sum())
    print(f"    Features: {n_features} selected "
          f"(var_thr={var_thr:.4f}, corr_thr={corr_thr:.4f})"
          + (f"; momentum baseline: {mom_name}" if mom_name else
             "; momentum baseline: NOT FOUND (skipped)"))

    # Val split fixed across seeds (partition depends on SEED+fold only)
    rng = np.random.RandomState(SEED + fold_id)
    n_total = len(X_tr_sel)
    perm = rng.permutation(n_total)
    n_val = int(n_total * 0.1)
    val_idx, fit_idx = perm[:n_val], perm[n_val:]
    X_fit = X_tr_sel[fit_idx]
    Yr_fit = Y_ret_tr_full[fit_idx]
    Yk_fit_log = Y_risk_tr_full_log[fit_idx]
    X_val = X_tr_sel[val_idx]
    Yr_val = Y_ret_tr_full[val_idx]
    Yk_val_log = Y_risk_tr_full_log[val_idx]
    print(f"    Train fit: {len(X_fit):,}, val: {len(X_val):,}, "
          f"test: {len(X_te_sel):,}")

    arch = getattr(config_module, 'TRAINING_NN_ARCHITECTURE', [64, 32, 16])
    if isinstance(arch, str):
        arch = {'small': [32, 16], 'medium': [64, 32, 16],
                'large': [128, 64, 32]}.get(arch, [64, 32, 16])
    lr = getattr(config_module, 'TRAINING_LR', 5e-4)
    weight_decay = getattr(config_module, 'TRAINING_WEIGHT_DECAY', 1e-4)
    epochs = getattr(config_module, 'TRAINING_EPOCHS', 5000)
    if epochs_cap:
        epochs = min(epochs, epochs_cap)
    patience = getattr(config_module, 'EARLY_STOP_PATIENCE', 41)
    dropout = getattr(config_module, 'TRAINING_DROPOUT', 0.2)
    batch_size = 256

    ens_ret_mu = []
    for nn_idx in range(n_ensemble):
        if seed_override is None:
            effective_seed = SEED + fold_id * 100 + nn_idx
        else:
            effective_seed = seed_override * 1000 + fold_id * 100 + nn_idx
        torch.manual_seed(effective_seed)
        np.random.seed(effective_seed)

        model = HeteroscedasticDualHeadNN(in_dim=n_features,
                                          hidden_dims=arch, dropout=dropout)
        optimizer = optim.Adam(model.parameters(), lr=lr,
                               weight_decay=weight_decay)

        X_fit_t = torch.tensor(X_fit, dtype=torch.float32)
        Yr_fit_t = torch.tensor(Yr_fit, dtype=torch.float32)
        Yk_fit_t = torch.tensor(Yk_fit_log, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        Yr_val_t = torch.tensor(Yr_val, dtype=torch.float32)
        Yk_val_t = torch.tensor(Yk_val_log, dtype=torch.float32)
        X_te_t = torch.tensor(X_te_sel, dtype=torch.float32)

        best_val = float('inf')
        best_epoch = 0
        best_state = None
        wait = 0
        for epoch in range(epochs):
            model.train()
            order = torch.randperm(len(X_fit_t))
            for i in range(0, len(order), batch_size):
                bi = order[i:i + batch_size]
                pred = model(X_fit_t[bi])
                loss, _, _ = LOSS_FN(pred, Yr_fit_t[bi], Yk_fit_t[bi])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                vl, _, _ = LOSS_FN(model(X_val_t), Yr_val_t, Yk_val_t)
            val_loss = vl.item()
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                best_state = {k: v.clone()
                              for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            ret_mu, _, _, _ = model(X_te_t)
        ens_ret_mu.append(ret_mu.numpy())
        print(f"    NN #{nn_idx+1}: val_NLL={best_val:.6f} at epoch "
              f"{best_epoch}, stopped at {epoch}")

    mean_pred = np.stack(ens_ret_mu).mean(axis=0)

    rows = per_date_metrics(te_dates, mean_pred, Y_ret_te, mom_signal)
    if len(rows) == 0:
        raise RuntimeError(f"Fold {fold_id}: no snapshot date reached "
                           f"MIN_DATE_COUNT={MIN_DATE_COUNT}")

    ics = np.array([r['ic'] for r in rows])
    alphas = np.array([r['alpha'] for r in rows])
    covered = sum(r['n'] for r in rows)
    fold_ic = float(ics.mean())
    fold_result = {
        'fold_id': fold_id,
        'test_year': test_year,
        'binding': binding,
        'purge_cutoff': purge_cutoff,
        'n_train': int(train_mask.sum()),
        'n_test_samples': int(test_mask.sum()),
        'n_dates_used': len(rows),
        'date_coverage': float(covered / test_mask.sum()),
        'ic': fold_ic,
        'ic_std_across_dates': float(ics.std()),
        'icir': float(fold_ic / ics.std()) if ics.std() > 0 else None,
        'ic_positive_date_frac': float((ics > 0).mean()),
        'alpha': float(alphas.mean()),
        'alpha_std_across_dates': float(alphas.std()),
        'mae': float(np.mean(np.abs(mean_pred - Y_ret_te))),
        'momentum_feature': mom_name,
        'momentum_ic': (float(np.mean([r['momentum_ic'] for r in rows]))
                        if mom_name else None),
        'n_features': n_features,
        'elapsed_min': (time.time() - t0) / 60,
    }

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(
            out_dir / f'fold{fold_id}_dates.csv', index=False)
    return fold_result, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed override for the 3-seed study {42,1,2}')
    parser.add_argument('--folds', default='0,1,2,3,4,5',
                        help='Comma-separated fold ids to run')
    parser.add_argument('--cache', default=CACHE)
    parser.add_argument('--smoke', action='store_true',
                        help='Tiny run for plumbing checks only: '
                             'N_ENSEMBLE=2, epoch cap 60. NOT for verdict.')
    parser.add_argument('--no-spy', action='store_true',
                        help='Skip SPY secondary (offline)')
    args = parser.parse_args()

    fold_ids = [int(x) for x in args.folds.split(',')]
    n_ensemble = 2 if args.smoke else getattr(config_module, 'N_ENSEMBLE', 20)
    epochs_cap = 60 if args.smoke else None

    seed_tag = f"seed{args.seed}" if args.seed is not None else "seeddefault"
    if args.smoke:
        seed_tag += "_smoke"
    out_dir = RESULTS_ROOT / seed_tag

    print("=" * 70)
    print(f"v2.3.17 PURGED WALK-FORWARD — {seed_tag}")
    print("=" * 70)
    print(f"Config: arch={getattr(config_module, 'TRAINING_NN_ARCHITECTURE')}, "
          f"lr={getattr(config_module, 'TRAINING_LR'):.6g}, "
          f"wd={getattr(config_module, 'TRAINING_WEIGHT_DECAY'):.6g}, "
          f"dropout={getattr(config_module, 'TRAINING_DROPOUT'):.4f}")
    print(f"N_ensemble={n_ensemble}, folds={fold_ids}, "
          f"purge={PURGE_CAL_DAYS} cal days (>= 63 trading days), "
          f"min_date_count={MIN_DATE_COUNT}")
    if args.smoke:
        print("*** SMOKE MODE — plumbing check only, not a verdict run ***")
    print()

    data = load_data(args.cache)

    fold_results = []
    all_rows = {}
    for fold_id, test_year, binding in FOLD_SPEC:
        if fold_id not in fold_ids:
            continue
        last_fold = (fold_id == 5)
        tag = 'binding' if binding else 'NON-BINDING'
        print("-" * 60)
        print(f"Fold {fold_id} — test {test_year}"
              f"{'+tail' if last_fold else ''} ({tag})")
        print("-" * 60)
        fr, rows = run_fold(data, fold_id, test_year, binding, last_fold,
                            args.seed, n_ensemble, epochs_cap, out_dir)
        print(f"  Fold {fold_id}: IC={fr['ic']:+.4f} "
              f"(ICIR={fr['icir']:.2f}, {fr['ic_positive_date_frac']*100:.0f}% "
              f"dates positive), alpha={fr['alpha']*100:+.2f}%p, "
              f"MAE={fr['mae']*100:.1f}%p, {fr['n_dates_used']} dates, "
              f"{fr['elapsed_min']:.1f} min\n")
        fold_results.append(fr)
        all_rows[fold_id] = rows

    # SPY secondary (non-binding)
    if not args.no_spy:
        try:
            dates_needed = sorted({r['date'] for rows in all_rows.values()
                                   for r in rows})
            spy_fwd = fetch_spy_forward(dates_needed)
            for fr in fold_results:
                rows = all_rows[fr['fold_id']]
                diffs = [r['top5_mean_ret'] - spy_fwd[r['date']]
                         for r in rows if r['date'] in spy_fwd]
                fr['alpha_vs_spy'] = (float(np.mean(diffs))
                                      if diffs else None)
                fr['spy_dates_matched'] = len(diffs)
            print(f"[SPY] matched {len(spy_fwd)} snapshot dates")
        except Exception as e:
            print(f"[SPY] skipped ({e})")

    # Aggregate over binding folds present in this run
    binding_frs = [fr for fr in fold_results if fr['binding']]
    agg = None
    if binding_frs:
        b_ics = [fr['ic'] for fr in binding_frs]
        agg = {
            'binding_folds': [fr['fold_id'] for fr in binding_frs],
            'mean_ic': float(np.mean(b_ics)),
            'fold_ics': b_ics,
            'n_positive_folds': int(sum(ic > 0 for ic in b_ics)),
            'mean_alpha': float(np.mean([fr['alpha'] for fr in binding_frs])),
            'fold_alphas': [fr['alpha'] for fr in binding_frs],
        }
        print("=" * 70)
        print(f"Binding aggregate ({len(binding_frs)} folds): "
              f"mean IC={agg['mean_ic']:+.4f}, "
              f"positive {agg['n_positive_folds']}/{len(binding_frs)}, "
              f"mean alpha={agg['mean_alpha']*100:+.2f}%p")
        print("(Verdict is computed across 3 seeds per pre-reg section 4 — "
              "this line is per-seed only.)")

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        'seed': args.seed,
        'smoke': bool(args.smoke),
        'n_ensemble': n_ensemble,
        'purge_cal_days': PURGE_CAL_DAYS,
        'min_date_count': MIN_DATE_COUNT,
        'config_used': {
            'architecture': str(getattr(config_module,
                                        'TRAINING_NN_ARCHITECTURE')),
            'lr': getattr(config_module, 'TRAINING_LR'),
            'weight_decay': getattr(config_module, 'TRAINING_WEIGHT_DECAY'),
            'dropout': getattr(config_module, 'TRAINING_DROPOUT'),
            'var_threshold': getattr(config_module, 'VAR_THRESHOLD'),
            'corr_threshold': getattr(config_module, 'CORR_THRESHOLD'),
            'epochs': getattr(config_module, 'TRAINING_EPOCHS'),
            'patience': getattr(config_module, 'EARLY_STOP_PATIENCE'),
        },
        'per_fold': fold_results,
        'binding_aggregate': agg,
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_dir}/summary.json")


if __name__ == '__main__':
    main()

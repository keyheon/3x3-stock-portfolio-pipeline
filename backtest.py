"""
backtest.py — Stratified K-Fold Portfolio Backtest.

Cross-sectional validation: split on the ticker axis and ask
"can the model pick profitable stocks among tickers it hasn't seen?"

Key questions:
  1. Selection Alpha: does the NN's top-5 actually beat the benchmark?
  2. Rank Correlation: do predicted return rankings match actual rankings?
  3. Cross-Sector Transfer: do patterns learned on tech also work on pharma?

Usage:
  python backtest.py                          # standalone
  python run.py --torch --screen --backtest   # run after the full pipeline

Design:
  - Stratified K-Fold: sectors distributed proportionally across folds
    (same time period, different stocks in each fold).
  - Why ticker-axis splits fit this task better than time-axis splits:
    The model's goal is "which feature patterns produce returns?" (ranking stocks),
    not "predict the future" (time-series forecasting).
"""

import numpy as np
import time
import os


def run_backtest(n_folds=5, n_select=5, verbose=True):
    """
    Stratified K-Fold Portfolio Backtest.

    Returns:
        dict with fold results, aggregate metrics, cross-sector diagnostics
    """
    if verbose:
        print("\n" + "=" * 70)
        print("STRATIFIED K-FOLD PORTFOLIO BACKTEST")
        print("  Cross-sectional validation: ticker-axis split")
        print("=" * 70)

    # ── Step 1: Build training data with ticker labels ──
    X, Y_ret, Y_risk, meta, feat_names = _load_or_build_data(verbose)

    # Extract ticker labels from meta
    sample_tickers = np.array([m[0] for m in meta])
    unique_tickers = sorted(set(sample_tickers))

    if verbose:
        print(f"\n  Data: {X.shape[0]:,} samples, {X.shape[1]} features")
        print(f"  Tickers: {len(unique_tickers)}")

    # ── Step 2: Get sector labels ──
    ticker_sectors = _get_ticker_sectors(unique_tickers, verbose)

    # ── Step 3: Create stratified folds ──
    folds = _stratified_kfold(unique_tickers, ticker_sectors, n_folds)

    if verbose:
        print(f"\n  {n_folds} Stratified Folds:")
        for i, (train_tk, test_tk) in enumerate(folds):
            train_n = sum(1 for t in sample_tickers if t in train_tk)
            test_n = sum(1 for t in sample_tickers if t in test_tk)
            print(f"    Fold {i+1}: train={len(train_tk)} tickers ({train_n:,} samples), "
                  f"test={len(test_tk)} tickers ({test_n:,} samples)")

    # ── Step 4: Run each fold ──
    fold_results = []
    for i, (train_tk, test_tk) in enumerate(folds):
        if verbose:
            print(f"\n{'─' * 50}")
            print(f"  Fold {i+1}/{n_folds}")

        result = _run_single_fold(
            X, Y_ret, Y_risk, sample_tickers,
            train_tk, test_tk, n_select, verbose)
        fold_results.append(result)

    # ── Step 5: Cross-sector diagnostic ──
    if verbose:
        print(f"\n{'─' * 50}")
        print(f"  Cross-Sector Transfer Diagnostic")

    cross_sector = _cross_sector_diagnostic(
        X, Y_ret, Y_risk, sample_tickers, ticker_sectors, verbose)

    # ── Step 6: Report ──
    report = _aggregate_and_report(fold_results, cross_sector, verbose)

    # Save results
    _save_results(report)

    return report


# ============================================================
# DATA LOADING (with cache)
# ============================================================

def _load_or_build_data(verbose=True):
    """Build training data or load from cache."""
    cache_path = os.path.join('results', 'backtest_cache.npz')

    if os.path.exists(cache_path):
        if verbose:
            print(f"\n  Loading cached data from {cache_path}...")
        data = np.load(cache_path, allow_pickle=True)
        X = data['X']
        Y_ret = data['Y_ret']
        Y_risk = data['Y_risk']
        meta = [tuple(m) for m in data['meta']]
        feat_names = list(data['feat_names'])
        if verbose:
            print(f"  Loaded: {X.shape[0]:,} samples × {X.shape[1]} features")
        return X, Y_ret, Y_risk, meta, feat_names

    if verbose:
        print(f"\n  Building training data (first run — will be cached)...")

    from historical import build_training_data, auto_expand_universe
    import config

    tickers = auto_expand_universe()
    X, Y_ret, Y_risk, meta, feat_names = build_training_data(
        tickers=tickers,
        period=getattr(config, 'TRAINING_PERIOD', '10y'),
        snapshot_interval=getattr(config, 'TRAINING_SNAPSHOT_INTERVAL', 10),
    )

    # Cache
    os.makedirs('results', exist_ok=True)
    np.savez_compressed(cache_path,
                        X=X, Y_ret=Y_ret, Y_risk=Y_risk,
                        meta=np.array(meta, dtype=object),
                        feat_names=np.array(feat_names))
    if verbose:
        print(f"  Cached to {cache_path}")

    return X, Y_ret, Y_risk, meta, feat_names


# ============================================================
# SECTOR LABELS
# ============================================================

def _get_ticker_sectors(tickers, verbose=True):
    """Map tickers to GICS sectors via Wikipedia; 'Unknown' on failure."""
    import pandas as pd
    import urllib.request
    from io import StringIO

    sector_map = {}

    try:
        headers = {"User-Agent": "Mozilla/5.0 (stock-pipeline/2.3)"}
        req = urllib.request.Request(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=headers)
        html = urllib.request.urlopen(req, timeout=15).read().decode()
        tables = pd.read_html(StringIO(html))
        sp500 = tables[0]
        for _, row in sp500.iterrows():
            tk = str(row.get('Symbol', '')).replace('.', '-').strip()
            sector = str(row.get('GICS Sector', 'Unknown'))
            sector_map[tk] = sector
    except:
        pass

    # Fill missing with 'Unknown'
    result = {}
    for tk in tickers:
        result[tk] = sector_map.get(tk, 'Unknown')

    if verbose:
        sector_counts = {}
        for s in result.values():
            sector_counts[s] = sector_counts.get(s, 0) + 1
        print(f"\n  GICS Sector Distribution:")
        for s, n in sorted(sector_counts.items(), key=lambda x: -x[1])[:8]:
            print(f"    {s}: {n}")

    return result


# ============================================================
# STRATIFIED K-FOLD
# ============================================================

def _stratified_kfold(tickers, ticker_sectors, n_folds):
    """K-fold split with sector-proportional stratification."""
    # Group tickers by sector
    sector_groups = {}
    for tk in tickers:
        sec = ticker_sectors.get(tk, 'Unknown')
        if sec not in sector_groups:
            sector_groups[sec] = []
        sector_groups[sec].append(tk)

    # Shuffle within each sector
    rng = np.random.RandomState(42)
    for sec in sector_groups:
        rng.shuffle(sector_groups[sec])

    # Distribute each sector's tickers across folds
    fold_tickers = [[] for _ in range(n_folds)]
    for sec, tks in sector_groups.items():
        for i, tk in enumerate(tks):
            fold_tickers[i % n_folds].append(tk)

    # Create train/test splits
    folds = []
    for i in range(n_folds):
        test_set = set(fold_tickers[i])
        train_set = set()
        for j in range(n_folds):
            if j != i:
                train_set.update(fold_tickers[j])
        folds.append((train_set, test_set))

    return folds


# ============================================================
# SINGLE FOLD EXECUTION
# ============================================================

def _run_single_fold(X, Y_ret, Y_risk, sample_tickers,
                     train_tickers, test_tickers, n_select, verbose):
    """Single fold: train -> predict -> evaluate."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import config

    # Split samples by ticker
    train_mask = np.array([t in train_tickers for t in sample_tickers])
    test_mask = np.array([t in test_tickers for t in sample_tickers])

    X_tr, Y_ret_tr, Y_risk_tr = X[train_mask], Y_ret[train_mask], Y_risk[train_mask]
    X_te, Y_ret_te, Y_risk_te = X[test_mask], Y_ret[test_mask], Y_risk[test_mask]
    test_tk_per_sample = sample_tickers[test_mask]

    if verbose:
        print(f"    Train: {X_tr.shape[0]:,} samples, Test: {X_te.shape[0]:,} samples")

    # Feature selection + normalization (on train only)
    var = np.var(X_tr, axis=0)
    keep = var > 0.01
    corr_with_ret = np.array([
        abs(np.corrcoef(X_tr[:, j], Y_ret_tr)[0, 1])
        if np.std(X_tr[:, j]) > 1e-10 else 0
        for j in range(X_tr.shape[1])
    ])
    keep = keep & (corr_with_ret > 0.05)

    X_tr_sel = X_tr[:, keep]
    X_te_sel = X_te[:, keep]

    mu = np.mean(X_tr_sel, axis=0)
    sigma = np.std(X_tr_sel, axis=0) + 1e-8
    X_tr_n = np.clip((X_tr_sel - mu) / sigma, -5, 5)
    X_te_n = np.clip((X_te_sel - mu) / sigma, -5, 5)

    if verbose:
        print(f"    Features: {X_tr_sel.shape[1]} selected")

    # Train ensemble (3 models for speed)
    n_ensemble = 3
    models = []
    D = X_tr_n.shape[1]

    for e in range(n_ensemble):
        torch.manual_seed(42 + e * 17)
        arch = getattr(config, 'TRAINING_NN_ARCHITECTURE', [64, 32, 16])
        layers = []
        in_dim = D
        for h in arch:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.2)])
            in_dim = h
        layers.append(nn.Linear(in_dim, 2))
        model = nn.Sequential(*layers)

        lr = getattr(config, 'TRAINING_LR', 0.0005)
        delta = getattr(config, 'TRAINING_HUBER_DELTA', 0.3)
        epochs = min(getattr(config, 'TRAINING_EPOCHS', 800), 500)  # capped for speed

        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        Xt = torch.tensor(X_tr_n, dtype=torch.float32)
        yr = torch.tensor(Y_ret_tr, dtype=torch.float32)
        yk = torch.tensor(Y_risk_tr, dtype=torch.float32)

        model.train()
        best_loss, patience = float('inf'), 0
        for ep in range(epochs):
            opt.zero_grad()
            out = model(Xt)
            loss = F.huber_loss(out[:, 0], yr, delta=delta) + \
                   F.huber_loss(F.softplus(out[:, 1]), yk, delta=delta)
            if torch.isnan(loss):
                break
            loss.backward()
            opt.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience = 0
            else:
                patience += 1
                if patience > 80:
                    break

        models.append(model)
        if verbose:
            print(f"    NN #{e+1}: loss={best_loss:.6f}, epochs={min(ep+1, epochs)}")

    # Predict on test samples
    X_te_t = torch.tensor(X_te_n, dtype=torch.float32)
    all_pred_ret = np.zeros(X_te_n.shape[0])
    all_pred_risk = np.zeros(X_te_n.shape[0])

    for model in models:
        model.eval()
        with torch.no_grad():
            out = model(X_te_t)
            all_pred_ret += out[:, 0].numpy()
            all_pred_risk += F.softplus(out[:, 1]).numpy()

    all_pred_ret /= n_ensemble
    all_pred_risk /= n_ensemble

    # Aggregate per ticker: mean prediction and mean actual
    test_tickers_list = sorted(test_tickers)
    ticker_pred_ret = {}
    ticker_pred_risk = {}
    ticker_actual_ret = {}
    ticker_actual_risk = {}
    ticker_n_samples = {}

    for i, tk in enumerate(test_tk_per_sample):
        if tk not in ticker_pred_ret:
            ticker_pred_ret[tk] = []
            ticker_pred_risk[tk] = []
            ticker_actual_ret[tk] = []
            ticker_actual_risk[tk] = []

        ticker_pred_ret[tk].append(all_pred_ret[i])
        ticker_pred_risk[tk].append(all_pred_risk[i])
        ticker_actual_ret[tk].append(Y_ret_te[i])
        ticker_actual_risk[tk].append(Y_risk_te[i])

    # Per-ticker averages
    tickers_eval = sorted(ticker_pred_ret.keys())
    pred_rets = np.array([np.mean(ticker_pred_ret[tk]) for tk in tickers_eval])
    pred_risks = np.array([np.mean(ticker_pred_risk[tk]) for tk in tickers_eval])
    actual_rets = np.array([np.mean(ticker_actual_ret[tk]) for tk in tickers_eval])
    actual_risks = np.array([np.mean(ticker_actual_risk[tk]) for tk in tickers_eval])
    n_samples_per = np.array([len(ticker_pred_ret[tk]) for tk in tickers_eval])

    # ── Metrics ──
    # 1. Rank Correlation
    from scipy.stats import spearmanr
    rank_corr, rank_p = spearmanr(pred_rets, actual_rets)
    if np.isnan(rank_corr):
        rank_corr = 0.0

    # 2. Selection Alpha
    if len(pred_rets) >= n_select:
        top_idx = np.argsort(pred_rets)[-n_select:]
        bottom_idx = np.argsort(pred_rets)[:n_select]

        top_actual = np.mean(actual_rets[top_idx])
        bottom_actual = np.mean(actual_rets[bottom_idx])
        all_actual = np.mean(actual_rets)
        selection_alpha = top_actual - all_actual
        long_short_spread = top_actual - bottom_actual
    else:
        top_actual = bottom_actual = all_actual = 0
        selection_alpha = long_short_spread = 0
        top_idx = np.array([])

    # 3. Hit Rate
    if len(top_idx) > 0:
        hit_rate = np.mean(actual_rets[top_idx] > 0) * 100
        beat_median = np.mean(actual_rets[top_idx] > np.median(actual_rets)) * 100
    else:
        hit_rate = beat_median = 0

    # 4. Sharpe of selection
    if len(top_idx) > 0 and np.std(actual_rets[top_idx]) > 0:
        select_sharpe = np.mean(actual_rets[top_idx]) / np.std(actual_rets[top_idx])
    else:
        select_sharpe = 0

    result = {
        'n_train_tickers': len(train_tickers),
        'n_test_tickers': len(tickers_eval),
        'n_train_samples': X_tr.shape[0],
        'n_test_samples': X_te.shape[0],
        'n_features': X_tr_sel.shape[1],
        'rank_corr': round(rank_corr, 4),
        'rank_p': round(rank_p, 6) if not np.isnan(rank_p) else 1.0,
        'selection_alpha': round(selection_alpha, 4),
        'long_short_spread': round(long_short_spread, 4),
        'top5_mean_return': round(top_actual, 4),
        'bottom5_mean_return': round(bottom_actual, 4),
        'all_mean_return': round(all_actual, 4),
        'hit_rate': round(hit_rate, 1),
        'beat_median': round(beat_median, 1),
        'select_sharpe': round(select_sharpe, 3),
        'top5_tickers': [tickers_eval[i] for i in top_idx] if len(top_idx) > 0 else [],
    }

    if verbose:
        print(f"    ── Results ──")
        print(f"    Rank Corr:      {rank_corr:+.3f} (p={rank_p:.4f})")
        print(f"    Selection Alpha: {selection_alpha*100:+.1f}%p (top5 vs all)")
        print(f"    Long-Short:     {long_short_spread*100:+.1f}%p (top5 vs bottom5)")
        print(f"    Hit Rate:       {hit_rate:.0f}% (top5 positive return)")
        print(f"    Beat Median:    {beat_median:.0f}% (top5 > median)")
        print(f"    Select Sharpe:  {select_sharpe:.3f}")
        if top_idx is not None and len(top_idx) > 0:
            top_names = [tickers_eval[i] for i in top_idx]
            print(f"    Top 5 picks:    {', '.join(top_names)}")

    return result


# ============================================================
# CROSS-SECTOR DIAGNOSTIC
# ============================================================

def _cross_sector_diagnostic(X, Y_ret, Y_risk, sample_tickers, ticker_sectors, verbose):
    """
    Cross-sector transfer test:
    Can patterns learned on tech also work on healthcare?

    Train on sector A -> test on sector B -> measure rank correlation.
    This diagnoses whether the model learned universal feature patterns
    or overfit to sector-specific behavior.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import config

    # Get broad sector groups (merge small groups)
    broad = {}
    for tk in set(sample_tickers):
        sec = ticker_sectors.get(tk, 'Unknown')
        broad[tk] = sec

    # Count samples per sector
    sector_counts = {}
    for tk in sample_tickers:
        s = broad.get(tk, 'Unknown')
        sector_counts[s] = sector_counts.get(s, 0) + 1

    # Keep sectors with enough samples (> 5000)
    large_sectors = [s for s, n in sector_counts.items()
                     if n > 5000 and s != 'Unknown']
    large_sectors = sorted(large_sectors)[:5]  # top 5 largest

    if verbose:
        print(f"    Testing cross-sector transfer between: {', '.join(large_sectors)}")

    results = []

    for train_sec in large_sectors:
        for test_sec in large_sectors:
            if train_sec == test_sec:
                continue

            # Get samples for each sector
            train_mask = np.array([broad.get(t, '') == train_sec for t in sample_tickers])
            test_mask = np.array([broad.get(t, '') == test_sec for t in sample_tickers])

            X_tr, Y_tr = X[train_mask], Y_ret[train_mask]
            X_te, Y_te = X[test_mask], Y_ret[test_mask]
            Yk_tr = Y_risk[train_mask]

            if len(X_tr) < 100 or len(X_te) < 100:
                continue

            # Quick feature selection + normalization
            var = np.var(X_tr, axis=0)
            keep = var > 0.01
            X_tr_s, X_te_s = X_tr[:, keep], X_te[:, keep]
            mu = np.mean(X_tr_s, axis=0)
            sigma = np.std(X_tr_s, axis=0) + 1e-8
            X_tr_n = np.clip((X_tr_s - mu) / sigma, -5, 5)
            X_te_n = np.clip((X_te_s - mu) / sigma, -5, 5)

            # Quick single-model training (200 epochs)
            D = X_tr_n.shape[1]
            torch.manual_seed(42)
            arch = getattr(config, 'TRAINING_NN_ARCHITECTURE', [64, 32, 16])
            layers = []
            in_dim = D
            for h in arch:
                layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.2)])
                in_dim = h
            layers.append(nn.Linear(in_dim, 2))
            model = nn.Sequential(*layers)

            delta = getattr(config, 'TRAINING_HUBER_DELTA', 0.3)
            opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
            Xt = torch.tensor(X_tr_n, dtype=torch.float32)
            yr = torch.tensor(Y_tr, dtype=torch.float32)
            yk = torch.tensor(Yk_tr, dtype=torch.float32)

            model.train()
            for ep in range(200):
                opt.zero_grad()
                out = model(Xt)
                loss = F.huber_loss(out[:, 0], yr, delta=delta) + \
                       F.huber_loss(F.softplus(out[:, 1]), yk, delta=delta)
                if torch.isnan(loss):
                    break
                loss.backward()
                opt.step()

            # Predict on test sector
            model.eval()
            with torch.no_grad():
                pred = model(torch.tensor(X_te_n, dtype=torch.float32))[:, 0].numpy()

            # Aggregate per ticker
            test_tickers_in = sorted(set(sample_tickers[test_mask]))
            tk_pred, tk_actual = {}, {}
            test_tk_arr = sample_tickers[test_mask]
            for i, tk in enumerate(test_tk_arr):
                if tk not in tk_pred:
                    tk_pred[tk] = []
                    tk_actual[tk] = []
                tk_pred[tk].append(pred[i])
                tk_actual[tk].append(Y_te[i])

            tks = sorted(tk_pred.keys())
            p_arr = np.array([np.mean(tk_pred[tk]) for tk in tks])
            a_arr = np.array([np.mean(tk_actual[tk]) for tk in tks])

            from scipy.stats import spearmanr
            corr, _ = spearmanr(p_arr, a_arr)
            if np.isnan(corr):
                corr = 0.0

            results.append({
                'train_sector': train_sec,
                'test_sector': test_sec,
                'rank_corr': round(corr, 3),
                'n_train': len(X_tr),
                'n_test': len(X_te),
                'n_test_tickers': len(tks),
            })

            if verbose:
                print(f"    {train_sec[:15]:>15} → {test_sec[:15]:<15}: "
                      f"rank_corr={corr:+.3f} ({len(tks)} tickers)")

    return results


# ============================================================
# AGGREGATE & REPORT
# ============================================================

def _aggregate_and_report(fold_results, cross_sector, verbose):
    """Aggregate fold results and produce final report."""
    n = len(fold_results)

    avg = {
        'rank_corr': np.mean([r['rank_corr'] for r in fold_results]),
        'selection_alpha': np.mean([r['selection_alpha'] for r in fold_results]),
        'long_short_spread': np.mean([r['long_short_spread'] for r in fold_results]),
        'hit_rate': np.mean([r['hit_rate'] for r in fold_results]),
        'beat_median': np.mean([r['beat_median'] for r in fold_results]),
        'select_sharpe': np.mean([r['select_sharpe'] for r in fold_results]),
        'top5_mean_return': np.mean([r['top5_mean_return'] for r in fold_results]),
        'all_mean_return': np.mean([r['all_mean_return'] for r in fold_results]),
    }

    std = {
        'rank_corr': np.std([r['rank_corr'] for r in fold_results]),
        'selection_alpha': np.std([r['selection_alpha'] for r in fold_results]),
        'long_short_spread': np.std([r['long_short_spread'] for r in fold_results]),
    }

    # Cross-sector average
    if cross_sector:
        within = [r['rank_corr'] for r in cross_sector
                  if r['train_sector'] == r.get('_same', None)]  # won't match
        cross = [r['rank_corr'] for r in cross_sector]
        avg_cross = np.mean(cross) if cross else 0
    else:
        avg_cross = 0

    report = {
        'n_folds': n,
        'fold_results': fold_results,
        'aggregate': avg,
        'std': std,
        'cross_sector': cross_sector,
        'cross_sector_avg_rank_corr': round(avg_cross, 3),
    }

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"BACKTEST RESULTS ({n}-Fold Stratified Cross-Sectional)")
        print(f"{'=' * 70}")
        print(f"")
        print(f"  Portfolio Selection Metrics (mean ± std across folds):")
        print(f"  {'─' * 55}")
        print(f"  Rank Correlation:   {avg['rank_corr']:+.3f} ± {std['rank_corr']:.3f}")
        print(f"  Selection Alpha:    {avg['selection_alpha']*100:+.1f}%p ± {std['selection_alpha']*100:.1f}%p")
        print(f"  Long-Short Spread:  {avg['long_short_spread']*100:+.1f}%p ± {std['long_short_spread']*100:.1f}%p")
        print(f"  Hit Rate:           {avg['hit_rate']:.0f}% (top-5 positive return)")
        print(f"  Beat Median:        {avg['beat_median']:.0f}% (top-5 > median)")
        print(f"  Selection Sharpe:   {avg['select_sharpe']:.3f}")
        print(f"  Top-5 Avg Return:   {avg['top5_mean_return']*100:+.1f}%")
        print(f"  Universe Avg Return:{avg['all_mean_return']*100:+.1f}%")
        print(f"")

        # Interpretation
        rc = avg['rank_corr']
        alpha = avg['selection_alpha']
        print(f"  Interpretation:")
        if rc > 0.3:
            print(f"    [OK] Rank Corr {rc:+.3f}: Strong - predicted rankings significantly align with actuals")
        elif rc > 0.15:
            print(f"    [~]  Rank Corr {rc:+.3f}: Moderate - weak ranking signal present")
        elif rc > 0:
            print(f"    [~]  Rank Corr {rc:+.3f}: Weak - minimal ranking power")
        else:
            print(f"    [X]  Rank Corr {rc:+.3f}: None - no ranking power")

        if alpha > 0.02:
            print(f"    [OK] Alpha {alpha*100:+.1f}%p: model selection generates excess return vs benchmark")
        elif alpha > 0:
            print(f"    [~]  Alpha {alpha*100:+.1f}%p: slight excess return (statistical significance uncertain)")
        else:
            print(f"    [X]  Alpha {alpha*100:+.1f}%p: model selection underperforms benchmark")

        if cross_sector:
            print(f"")
            print(f"  Cross-Sector Transfer:")
            print(f"    Avg rank corr: {avg_cross:+.3f}")
            if avg_cross > 0.15:
                print(f"    [OK] Universal patterns - cross-sector transfer works")
            elif avg_cross > 0:
                print(f"    [~]  Partial transfer - only some patterns transfer")
            else:
                print(f"    [X]  Sector-specific - patterns are sector-dependent")

        print(f"\n{'=' * 70}")

    return report


def _save_results(report):
    """Save backtest results as JSON."""
    import json
    os.makedirs('results', exist_ok=True)

    # Clean for JSON serialization
    save = {
        'n_folds': report['n_folds'],
        'aggregate': report['aggregate'],
        'std': report['std'],
        'cross_sector_avg_rank_corr': report['cross_sector_avg_rank_corr'],
        'fold_results': report['fold_results'],
        'cross_sector': report['cross_sector'],
    }

    path = os.path.join('results', 'backtest_results.json')
    with open(path, 'w') as f:
        json.dump(save, f, indent=2, default=str)
    print(f"  Saved to {path}")


# ============================================================
# STANDALONE
# ============================================================

if __name__ == "__main__":
    run_backtest(n_folds=5, n_select=5, verbose=True)

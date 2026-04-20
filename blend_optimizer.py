"""
blend_optimizer.py — data-driven Stage 2 weight optimization.

Approach:
  1. Multi-window backtest (3m, 6m, 9m) to avoid overfitting to a single regime.
  2. Regime detection — if analyst rank is strongly negative, add shrinkage.
  3. Bounded weights: w_ret in [0.15, 0.60] to avoid extremes.
  4. Prior: w_ret = 0.30 (reflects the v1 experience where tech ~0.20 gave good Sharpe).

Why the analyst signal can look bad in a single window:
  - Analyst target = 12-month forward forecast
  - Backtest = past 3-month realized returns
  - The time-axis mismatch plus mean reversion means recently-fallen stocks
    have the largest upside, producing a negative Spearman with recent actuals.
  - This is structural, not a bug — multi-window + shrinkage handles it.
"""

import numpy as np
import time


def _spearman_corr(x, y):
    """Spearman rank correlation."""
    n = len(x)
    if n < 3:
        return 0.0
    try:
        from scipy.stats import spearmanr
        corr, _ = spearmanr(x, y)
        return corr if not np.isnan(corr) else 0.0
    except ImportError:
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        d = rx - ry
        return 1 - 6 * np.sum(d**2) / (n * (n**2 - 1))


def _backtest_sharpe(predicted_returns, actual_returns, n_select=5):
    """Sharpe of actual returns on top-N predicted selections."""
    if len(predicted_returns) < n_select:
        return 0.0
    top_idx = np.argsort(predicted_returns)[-n_select:]
    selected_actual = actual_returns[top_idx]
    mean_ret = np.mean(selected_actual)
    std_ret = np.std(selected_actual) + 1e-8
    return mean_ret / std_ret


def _run_single_window_backtest(tickers, trained_models, keep_mask, feat_mu, feat_sigma,
                                 analyst_targets, realized_risks, hist_X_shape_1,
                                 backtest_days):
    """Collect backtest data for a single window."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import yfinance as yf
    from historical import _SliceFrame
    from data_auto import compute_technical_features

    tech_bt, nn_risk_bt, actual_ret, actual_risk = {}, {}, {}, {}
    valid = []

    for tk in tickers:
        try:
            hist = yf.Ticker(tk).history(period="2y")
            hist = hist.dropna(subset=['Close'])
            close = hist['Close'].values
            if len(close) < backtest_days + 63:
                continue

            cutoff = len(close) - backtest_days
            cp = close[:cutoff]
            hp = hist['High'].values[:cutoff]
            lp = hist['Low'].values[:cutoff]
            vp = hist['Volume'].values[:cutoff]
            if len(cp) < 63:
                continue

            sf = _SliceFrame(cp, hp, lp, vp)
            feats = compute_technical_features(sf)
            fn = sorted(feats.keys())
            vec = np.nan_to_num(np.array([feats.get(k, 0) for k in fn], dtype=float))

            if len(vec) < hist_X_shape_1:
                vec = np.concatenate([vec, np.zeros(hist_X_shape_1 - len(vec))])
            elif len(vec) > hist_X_shape_1:
                vec = vec[:hist_X_shape_1]

            vec_sel = vec[keep_mask]
            vec_n = np.clip((vec_sel - feat_mu) / feat_sigma, -5, 5)
            x_t = torch.tensor(vec_n, dtype=torch.float32).unsqueeze(0)

            rets, risks = [], []
            for model in trained_models:
                model.eval()
                for m in model.modules():
                    if isinstance(m, nn.Dropout):
                        m.train()
                for _ in range(6):
                    with torch.no_grad():
                        out = model(x_t)
                        rets.append(out[0, 0].item())
                        risks.append(F.softplus(out[0, 1]).item())

            tech_bt[tk] = np.mean(rets)
            nn_risk_bt[tk] = np.mean(risks)
            actual_ret[tk] = (close[-1] / close[cutoff] - 1)
            fwd = np.diff(close[cutoff:]) / close[cutoff:-1]
            actual_risk[tk] = np.std(fwd) * np.sqrt(252) if len(fwd) > 5 else 0.3
            valid.append(tk)
        except:
            continue
        time.sleep(0.03)

    return valid, tech_bt, nn_risk_bt, actual_ret, actual_risk


def optimize_stage2_weights(tickers, trained_models, keep_mask, feat_mu, feat_sigma,
                            analyst_targets, realized_risks,
                            hist_X_shape_1, backtest_days=63, verbose=True):
    """
    Multi-window backtest + composite metric + regime detection
    to find optimal Stage 2 weights.
    """
    if verbose:
        print(f"\n  [Stage 2 Optimization] Multi-window + Regime Detection (v2.2)")

    # Step 1: multi-window backtest
    windows = [63, 126, 189]  # 3m, 6m, 9m
    window_results = []

    for bdays in windows:
        if verbose:
            print(f"    Window {bdays}d (~{bdays//21}m): ", end="")

        valid, tech_bt, nn_risk_bt, actual_ret, actual_risk = \
            _run_single_window_backtest(
                tickers, trained_models, keep_mask, feat_mu, feat_sigma,
                analyst_targets, realized_risks, hist_X_shape_1, bdays)

        if len(valid) < 10:
            if verbose:
                print(f"skip ({len(valid)} stocks)")
            continue

        tech_arr = np.array([tech_bt[tk] for tk in valid])
        analyst_arr = np.array([analyst_targets.get(tk, 0) for tk in valid])
        actual_arr = np.array([actual_ret[tk] for tk in valid])
        nn_risk_arr = np.array([nn_risk_bt[tk] for tk in valid])
        real_risk_arr = np.array([realized_risks.get(tk, 0.3) for tk in valid])
        act_risk_arr = np.array([actual_risk[tk] for tk in valid])

        # Diagnostics for this window
        tech_rank = _spearman_corr(tech_arr, actual_arr)
        analyst_rank = _spearman_corr(analyst_arr, actual_arr)
        tech_sharpe = _backtest_sharpe(tech_arr, actual_arr)
        analyst_sharpe = _backtest_sharpe(analyst_arr, actual_arr)

        if verbose:
            print(f"{len(valid)} stocks | Tech rank={tech_rank:+.2f} Sharpe={tech_sharpe:+.2f} | "
                  f"Analyst rank={analyst_rank:+.2f} Sharpe={analyst_sharpe:+.2f}")

        # Grid search for this window
        best_wr, best_wk, _ = _composite_grid_search(
            tech_arr, analyst_arr, actual_arr,
            nn_risk_arr, real_risk_arr, act_risk_arr,
            w_range=np.arange(0.15, 0.61, 0.05))

        window_results.append({
            'days': bdays, 'n': len(valid),
            'w_ret': best_wr, 'w_risk': best_wk,
            'tech_rank': tech_rank, 'analyst_rank': analyst_rank,
            'tech_sharpe': tech_sharpe, 'analyst_sharpe': analyst_sharpe,
        })

    if not window_results:
        if verbose:
            print(f"    [WARNING] No valid windows. Using default.")
        return _default_result()

    # Step 2: aggregate multi-window results
    # recency bias — recent window gets more weight
    weights_map = {63: 0.5, 126: 0.3, 189: 0.2}
    total_weight = sum(weights_map.get(r['days'], 0.2) for r in window_results)

    avg_w_ret = sum(r['w_ret'] * weights_map.get(r['days'], 0.2) for r in window_results) / total_weight
    avg_w_risk = sum(r['w_risk'] * weights_map.get(r['days'], 0.2) for r in window_results) / total_weight

    if verbose:
        print(f"    Multi-window weighted avg: w_ret={avg_w_ret:.2f}, w_risk={avg_w_risk:.2f}")

    # Step 3: regime detection
    # regime unstable if any window has strongly negative analyst rank
    analyst_ranks = [r['analyst_rank'] for r in window_results]
    regime_unstable = any(r < -0.3 for r in analyst_ranks)
    regime_positive = all(r > 0.1 for r in analyst_ranks)

    if regime_unstable:
        if verbose:
            print(f"    [!] Regime: analyst rank negative in some windows -> extra shrinkage")
        shrinkage = 0.5  # 50% shrinkage toward prior
    elif regime_positive:
        if verbose:
            print(f"    [OK] Regime: analyst consistently positive -> minimal shrinkage")
        shrinkage = 0.15
    else:
        shrinkage = 0.3

    # Prior based on v1 experience: 0.20 tech worked well
    prior_w_ret = 0.30
    prior_w_risk = 0.40

    final_w_ret = round((1 - shrinkage) * avg_w_ret + shrinkage * prior_w_ret, 2)
    final_w_risk = round((1 - shrinkage) * avg_w_risk + shrinkage * prior_w_risk, 2)

    # Bounds: [0.15, 0.60] for return, [0.05, 0.70] for risk
    final_w_ret = max(0.15, min(0.60, final_w_ret))
    final_w_risk = max(0.05, min(0.70, final_w_risk))

    # ── Step 4: Compare old vs optimized ──
    # Use the shortest window for comparison stats
    w0 = window_results[0]
    valid = []
    for tk in tickers:
        try:
            if tk in _run_single_window_backtest.__code__.co_varnames:
                pass
        except:
            pass

    # Re-run comparison on 3m window data
    valid_3m, tech_bt, _, actual_ret_3m, _ = \
        _run_single_window_backtest(
            tickers[:20], trained_models, keep_mask, feat_mu, feat_sigma,  # subset for speed
            analyst_targets, realized_risks, hist_X_shape_1, 63)

    if len(valid_3m) >= 5:
        t_arr = np.array([tech_bt[tk] for tk in valid_3m])
        a_arr = np.array([analyst_targets.get(tk, 0) for tk in valid_3m])
        act_arr = np.array([actual_ret_3m[tk] for tk in valid_3m])

        old_blend = 0.2 * t_arr + 0.8 * a_arr
        opt_blend = final_w_ret * t_arr + (1 - final_w_ret) * a_arr
        old_rank = _spearman_corr(old_blend, act_arr)
        opt_rank = _spearman_corr(opt_blend, act_arr)
        old_sharpe = _backtest_sharpe(old_blend, act_arr)
        opt_sharpe = _backtest_sharpe(opt_blend, act_arr)
    else:
        old_rank = opt_rank = old_sharpe = opt_sharpe = 0.0

    if verbose:
        print(f"\n    [Results]")
        print(f"    Optimal w_ret:  {final_w_ret:.2f} (tech) / {1-final_w_ret:.2f} (fund)")
        print(f"    Optimal w_risk: {final_w_risk:.2f} (NN) / {1-final_w_risk:.2f} (realized)")
        print(f"    Shrinkage:      {shrinkage:.0%} ({'regime unstable' if regime_unstable else 'normal'})")
        print(f"    Prior:          w_ret={prior_w_ret}, w_risk={prior_w_risk}")
        for r in window_results:
            print(f"    Window {r['days']:3d}d:   w_ret={r['w_ret']:.2f}  analyst_rank={r['analyst_rank']:+.2f}")

    return {
        'w_ret': final_w_ret,
        'w_risk': final_w_risk,
        'w_ret_raw': round(avg_w_ret, 2),
        'w_risk_raw': round(avg_w_risk, 2),
        'shrinkage': round(shrinkage, 2),
        'regime': 'unstable' if regime_unstable else ('positive' if regime_positive else 'mixed'),
        'windows': window_results,
        'n_windows': len(window_results),
        'method': 'multi-window(3m/6m/9m) + composite + regime detection + bounded shrinkage',
    }


def _composite_grid_search(tech_ret, analyst_ret, actual_ret,
                           nn_risk, realized_risk, actual_risk,
                           w_range=None, w_range_ret=None, w_range_risk=None,
                           n_select=5):
    """Composite metric grid search within bounds."""
    if w_range_ret is None:
        w_range_ret = w_range
    if w_range_risk is None:
        w_range_risk = w_range if w_range is not None else np.arange(0.05, 0.71, 0.05)

    best_w_ret, best_w_risk = 0.30, 0.40
    best_score = float('inf')
    baseline_mae = max(np.mean(np.abs(actual_ret)), 0.05)

    # Weights for composite metric
    lam_rank = 3.0
    lam_sharpe = 2.0
    lam_mae = 1.0

    for wr in w_range_ret:
        blend = wr * tech_ret + (1 - wr) * analyst_ret
        rank = _spearman_corr(blend, actual_ret)
        sharpe = _backtest_sharpe(blend, actual_ret, n_select)
        mae = np.mean(np.abs(blend - actual_ret))

        ret_score = -lam_rank * rank - lam_sharpe * sharpe + lam_mae * (mae / baseline_mae)

        for wk in w_range_risk:
            blend_risk = wk * nn_risk + (1 - wk) * realized_risk
            risk_mae = np.mean(np.abs(blend_risk - actual_risk))
            total = ret_score + risk_mae

            if total < best_score:
                best_score = total
                best_w_ret = wr
                best_w_risk = wk

    return round(best_w_ret, 3), round(best_w_risk, 3), best_score


def _default_result():
    return {
        'w_ret': 0.25, 'w_risk': 0.40,
        'w_ret_raw': 0.25, 'w_risk_raw': 0.40,
        'shrinkage': 0.5, 'regime': 'unknown',
        'windows': [], 'n_windows': 0,
        'method': 'default (insufficient data)',
    }


if __name__ == "__main__":
    print("blend_optimizer.py v2.2")
    print("Multi-window + Regime Detection + Bounded Shrinkage")
    print("Use: python run.py --torch --screen --sent")

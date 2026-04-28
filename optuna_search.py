"""
optuna_search.py — Stage 1 hyperparameter search (Task #6).

6-dim search: lr, weight_decay, huber_delta, architecture, var_thr, corr_thr.
N=5 ensemble, 4 folds (Fold 2-5; Fold 1 excluded due to SNDK artifact),
60 trials. Cache reuse via results/backtest_cache.npz. Resume-safe (sqlite).

Usage:
  caffeinate -i python optuna_search.py 2>&1 | tee optuna_stage1.log
"""

import sys
import os
import json
import time
import signal
import traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# STAGE 1 CONFIGURATION
# ============================================================

STUDY_NAME = "stage1_nn_feat_6dims"
STORAGE = "sqlite:///optuna_storage.db"
N_TRIALS = 60
N_ENSEMBLE_STAGE1 = 5
N_SELECT = 5
N_FOLDS_TOTAL = 5

# Fold 1 (idx 0) excluded — SNDK post-IPO artifact (v2.3.3 §32.4, v2.3.4 §39.6)
STAGE1_FOLD_INDICES = [1, 2, 3, 4]

TRIAL_TIMEOUT_SEC = 120 * 60
CACHE_PATH = "results/backtest_cache.npz"
RESULTS_PATH = "results/optuna_stage1_results.json"

SAMPLER_SEED = 42
N_STARTUP_TRIALS = 10

ARCH_CHOICES = {
    "small":  [32, 16],
    "medium": [64, 32, 16],
    "large":  [128, 64, 32],
}


# ============================================================
# TIMEOUT HANDLING
# ============================================================

class TrialTimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TrialTimeoutError(f"Trial exceeded {TRIAL_TIMEOUT_SEC}s")


# ============================================================
# CONFIG OVERRIDE (with restoration)
# ============================================================

def _override_config(config_module, overrides):
    """Override config values; return originals for restoration."""
    originals = {}
    for key, value in overrides.items():
        originals[key] = getattr(config_module, key, None)
        setattr(config_module, key, value)
    return originals


def _restore_config(config_module, originals):
    for key, value in originals.items():
        setattr(config_module, key, value)


# ============================================================
# DATA LOADING (cached, reused across trials)
# ============================================================

_DATA_CACHE = {}
_FOLDS_CACHE = None


def _load_data():
    """Load cached training data. Fail loudly if cache missing."""
    if 'X' in _DATA_CACHE:
        return _DATA_CACHE

    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError(
            f"Cache not found: {CACHE_PATH}\n"
            f"Run `python backtest.py` first to build it."
        )

    print(f"[Data] Loading {CACHE_PATH}...")
    data = np.load(CACHE_PATH, allow_pickle=True)
    X = data['X']
    Y_ret = data['Y_ret']
    Y_risk = data['Y_risk']
    meta = [tuple(m) for m in data['meta']]
    feat_names = list(data['feat_names'])

    sample_tickers = np.array([m[0] for m in meta])
    unique_tickers = sorted(set(sample_tickers))

    _DATA_CACHE.update({
        'X': X, 'Y_ret': Y_ret, 'Y_risk': Y_risk,
        'meta': meta, 'feat_names': feat_names,
        'sample_tickers': sample_tickers,
        'unique_tickers': unique_tickers,
    })
    print(f"[Data] {X.shape[0]:,} samples × {X.shape[1]} features, "
          f"{len(unique_tickers)} tickers")
    return _DATA_CACHE


def _get_folds():
    """Stratified K-fold splits (computed once, reused)."""
    global _FOLDS_CACHE
    if _FOLDS_CACHE is not None:
        return _FOLDS_CACHE

    from backtest import _stratified_kfold, _get_ticker_sectors

    data = _load_data()
    ticker_sectors = _get_ticker_sectors(data['unique_tickers'], verbose=False)
    folds = _stratified_kfold(data['unique_tickers'], ticker_sectors, N_FOLDS_TOTAL)

    print(f"[Folds] Using indices {STAGE1_FOLD_INDICES} (Fold 2-5)")
    _FOLDS_CACHE = folds
    return folds


# ============================================================
# OBJECTIVE FUNCTION
# ============================================================

def objective(trial):
    """Run a single trial; return mean rank_corr across Fold 2-5."""
    trial_start = time.time()
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(TRIAL_TIMEOUT_SEC)

    try:
        # 1. Suggest hyperparameters (6 dims)
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        huber_delta = trial.suggest_categorical("huber_delta", [0.1, 0.2, 0.3, 0.5, 1.0])
        arch_choice = trial.suggest_categorical("architecture", list(ARCH_CHOICES.keys()))
        var_threshold = trial.suggest_float("var_threshold", 1e-3, 1e-1, log=True)
        corr_threshold = trial.suggest_float("corr_threshold", 1e-3, 1e-1, log=True)
        architecture = ARCH_CHOICES[arch_choice]

        print(f"\n{'='*60}")
        print(f"[Trial {trial.number}] {time.strftime('%H:%M:%S')}")
        print(f"  lr={lr:.5f}  wd={weight_decay:.5f}  huber={huber_delta}")
        print(f"  arch={arch_choice} {architecture}")
        print(f"  var_thr={var_threshold:.4f}  corr_thr={corr_threshold:.4f}")
        print(f"{'='*60}")

        # 2. Override config
        import config
        overrides = {
            'TRAINING_LR': lr,
            'TRAINING_NN_ARCHITECTURE': architecture,
            'TRAINING_HUBER_DELTA': huber_delta,
            'VAR_THRESHOLD': var_threshold,
            'CORR_THRESHOLD': corr_threshold,
            'N_ENSEMBLE': N_ENSEMBLE_STAGE1,
        }
        originals = _override_config(config, overrides)

        try:
            data = _load_data()
            folds = _get_folds()

            # 3. Patch Adam default weight_decay (backtest.py hardcodes 1e-4)
            import torch.optim
            _original_adam_init = torch.optim.Adam.__init__
            _wd_target = weight_decay

            def _patched_adam_init(self, params, lr=0.001, betas=(0.9, 0.999),
                                   eps=1e-8, weight_decay=None, amsgrad=False, **kw):
                if weight_decay is None or weight_decay == 1e-4:
                    weight_decay = _wd_target
                return _original_adam_init(self, params, lr=lr, betas=betas,
                                           eps=eps, weight_decay=weight_decay,
                                           amsgrad=amsgrad, **kw)

            torch.optim.Adam.__init__ = _patched_adam_init

            try:
                # 4. Run selected folds
                from backtest import _run_single_fold

                fold_rank_corrs = []
                fold_alphas = []

                for fold_idx in STAGE1_FOLD_INDICES:
                    train_tickers, test_tickers = folds[fold_idx]
                    fold_start = time.time()
                    print(f"\n  --- Fold {fold_idx+1}/5 ---")

                    result = _run_single_fold(
                        data['X'], data['Y_ret'], data['Y_risk'],
                        data['sample_tickers'], data['meta'],
                        train_tickers, test_tickers, N_SELECT,
                        verbose=True,
                    )
                    rc = result['rank_corr']
                    alpha = result['selection_alpha']
                    fold_rank_corrs.append(rc)
                    fold_alphas.append(alpha)

                    print(f"  Fold {fold_idx+1}: rank_corr={rc:+.3f}  "
                          f"alpha={alpha*100:+.1f}%p  "
                          f"elapsed={(time.time()-fold_start)/60:.1f}min")

                mean_rank_corr = float(np.mean(fold_rank_corrs))
                mean_alpha = float(np.mean(fold_alphas))

                trial.set_user_attr("fold_rank_corrs",
                                    [float(v) for v in fold_rank_corrs])
                trial.set_user_attr("fold_alphas",
                                    [float(v) for v in fold_alphas])
                trial.set_user_attr("mean_alpha", mean_alpha)
                trial.set_user_attr("elapsed_sec", time.time() - trial_start)

                print(f"\n[Trial {trial.number}] rank_corr={mean_rank_corr:+.4f}  "
                      f"alpha={mean_alpha*100:+.1f}%p  "
                      f"elapsed={(time.time()-trial_start)/60:.1f}min")

                return mean_rank_corr

            finally:
                torch.optim.Adam.__init__ = _original_adam_init

        finally:
            _restore_config(config, originals)

    except TrialTimeoutError:
        print(f"\n[Trial {trial.number}] TIMEOUT ({TRIAL_TIMEOUT_SEC/60:.0f}min)")
        trial.set_user_attr("timeout", True)
        import optuna
        raise optuna.TrialPruned()

    except Exception as e:
        print(f"\n[Trial {trial.number}] ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        trial.set_user_attr("error", str(e))
        import optuna
        raise optuna.TrialPruned()

    finally:
        signal.alarm(0)


# ============================================================
# MAIN
# ============================================================

def main():
    import optuna

    print("="*70)
    print("OPTUNA STAGE 1 — 6-dim hyperparameter search")
    print("="*70)
    print(f"Study:     {STUDY_NAME}")
    print(f"Storage:   {STORAGE}")
    print(f"Trials:    {N_TRIALS}  (ensemble N={N_ENSEMBLE_STAGE1}, "
          f"folds={STAGE1_FOLD_INDICES})")
    print(f"Timeout:   {TRIAL_TIMEOUT_SEC/60:.0f}min per trial")
    print("="*70)

    if not os.path.exists(CACHE_PATH):
        print(f"\nERROR: {CACHE_PATH} not found. Run `python backtest.py` first.")
        sys.exit(1)

    sampler = optuna.samplers.TPESampler(
        seed=SAMPLER_SEED,
        n_startup_trials=N_STARTUP_TRIALS,
    )
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )

    completed = len([t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"\n[Study] {len(study.trials)} trials, {completed} completed")

    if completed > 0:
        best_trials = [t for t in study.trials
                       if t.state == optuna.trial.TrialState.COMPLETE]
        if best_trials:
            best = max(best_trials, key=lambda t: t.value or -1)
            print(f"[Study] Best: rank_corr={best.value:+.4f}  "
                  f"trial #{best.number}")

    remaining = max(0, N_TRIALS - completed)
    if remaining == 0:
        print(f"[Study] Target reached ({N_TRIALS}). Nothing to do.")
    else:
        print(f"[Study] Running {remaining} more trials...")
        study.optimize(
            objective,
            n_trials=remaining,
            catch=(Exception,),
            show_progress_bar=False,
        )

    # Save top-3 configs for Stage 2
    completed_trials = [t for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE]
    top_trials = sorted(completed_trials,
                        key=lambda t: t.value or -1, reverse=True)[:3]

    top_configs = []
    for i, t in enumerate(top_trials):
        top_configs.append({
            "rank": i + 1,
            "trial_number": t.number,
            "mean_rank_corr": float(t.value),
            "mean_alpha": t.user_attrs.get("mean_alpha"),
            "fold_rank_corrs": t.user_attrs.get("fold_rank_corrs"),
            "fold_alphas": t.user_attrs.get("fold_alphas"),
            "elapsed_sec": t.user_attrs.get("elapsed_sec"),
            "params": t.params,
        })

    os.makedirs("results", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "study_name": STUDY_NAME,
            "n_trials_total": len(study.trials),
            "n_trials_completed": len(completed_trials),
            "n_trials_target": N_TRIALS,
            "best_mean_rank_corr": float(top_trials[0].value) if top_trials else None,
            "top_3_configs": top_configs,
            "stage1_settings": {
                "n_ensemble": N_ENSEMBLE_STAGE1,
                "fold_indices": STAGE1_FOLD_INDICES,
                "timeout_sec": TRIAL_TIMEOUT_SEC,
                "sampler_seed": SAMPLER_SEED,
            },
        }, f, indent=2)

    print(f"\n{'='*70}")
    print(f"STAGE 1 COMPLETE: {len(completed_trials)}/{N_TRIALS} trials")
    print(f"Results: {RESULTS_PATH}")
    print(f"{'='*70}")
    for c in top_configs:
        print(f"  #{c['rank']} rank_corr={c['mean_rank_corr']:+.4f}  "
              f"alpha={c['mean_alpha']*100:+.1f}%p  {c['params']}")


if __name__ == "__main__":
    main()

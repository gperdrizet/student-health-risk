"""GPU-parallel fold scoring for the metalearner notebook."""
from __future__ import annotations

import os
import sys
from multiprocessing import get_context


def _worker(gpu_id, fold_batch, params, fit_seed, use_sampling,
            sample_train_frac, sample_val_frac, use_balanced, sample_seed, output_queue):
    """Fit + score one batch of meta-folds on a single GPU."""
    # Must be set before any CUDA initialisation in this worker process.
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Fallback path setup in case PYTHONPATH did not propagate.
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    import numpy as np
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.utils.class_weight import compute_sample_weight
    from hill_climbing_ensemble.ml_utils import XGBoostError, build_xgb_model

    results = []
    for fold_idx, fold in fold_batch:
        x_tr = fold['x_train']
        y_tr = fold['y_train']
        x_va = fold['x_validation']
        y_va = fold['y_validation']

        if use_sampling:
            rng = np.random.default_rng(sample_seed + fold_idx)
            tr_idx = rng.choice(len(y_tr), size=max(1, int(len(y_tr) * sample_train_frac)), replace=False)
            va_idx = rng.choice(len(y_va), size=max(1, int(len(y_va) * sample_val_frac)), replace=False)
            x_tr, y_tr = x_tr[tr_idx], y_tr[tr_idx]
            x_va, y_va = x_va[va_idx], y_va[va_idx]

        sw = compute_sample_weight('balanced', y=y_tr) if use_balanced else None
        model = build_xgb_model(params, seed=fit_seed + fold_idx, prefer_gpu=True)
        try:
            model.fit(x_tr, y_tr, sample_weight=sw)
        except XGBoostError:
            model = build_xgb_model(params, seed=fit_seed + fold_idx, prefer_gpu=False)
            model.fit(x_tr, y_tr, sample_weight=sw)

        results.append((fold_idx, float(balanced_accuracy_score(y_va, model.predict(x_va)))))

    output_queue.put(results)


def score_meta_folds_parallel(folds, params, fit_seed, gpu_ids, use_sampling=False,
                               sample_train_frac=1.0, sample_val_frac=1.0,
                               use_balanced=True, sample_seed=315):
    """Score metalearner params across folds, splitting folds across GPUs.

    Falls back to sequential single-GPU scoring when only one GPU is available.
    Uses spawn context to avoid CUDA context inheritance issues after fork.
    """
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.utils.class_weight import compute_sample_weight

    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from hill_climbing_ensemble.ml_utils import XGBoostError, build_xgb_model

    if len(gpu_ids) <= 1 or len(folds) <= 1:
        scores = []
        for fold_idx, fold in enumerate(folds, start=1):
            x_tr, y_tr = fold['x_train'], fold['y_train']
            x_va, y_va = fold['x_validation'], fold['y_validation']
            if use_sampling:
                rng = np.random.default_rng(sample_seed + fold_idx)
                tr_idx = rng.choice(len(y_tr), size=max(1, int(len(y_tr) * sample_train_frac)), replace=False)
                va_idx = rng.choice(len(y_va), size=max(1, int(len(y_va) * sample_val_frac)), replace=False)
                x_tr, y_tr = x_tr[tr_idx], y_tr[tr_idx]
                x_va, y_va = x_va[va_idx], y_va[va_idx]
            sw = compute_sample_weight('balanced', y=y_tr) if use_balanced else None
            model = build_xgb_model(params, seed=fit_seed + fold_idx, prefer_gpu=True)
            try:
                model.fit(x_tr, y_tr, sample_weight=sw)
            except XGBoostError:
                model = build_xgb_model(params, seed=fit_seed + fold_idx, prefer_gpu=False)
                model.fit(x_tr, y_tr, sample_weight=sw)
            scores.append(float(balanced_accuracy_score(y_va, model.predict(x_va))))
        return scores

    # Split folds round-robin across available GPUs.
    indexed = list(enumerate(folds, start=1))
    batches = [indexed[i::len(gpu_ids)] for i in range(len(gpu_ids))]

    ctx = get_context('spawn')
    q = ctx.Queue()
    procs = []

    for gpu_id, batch in zip(gpu_ids, batches):
        if not batch:
            continue
        p = ctx.Process(
            target=_worker,
            args=(gpu_id, batch, params, fit_seed, use_sampling,
                  sample_train_frac, sample_val_frac, use_balanced, sample_seed, q),
        )
        p.start()
        procs.append(p)

    all_results = []
    for _ in procs:
        all_results.extend(q.get())
    for p in procs:
        p.join()
        if p.exitcode not in (0, None):
            raise RuntimeError(f'Parallel fold worker exited with code {p.exitcode}')

    return [score for _, score in sorted(all_results)]

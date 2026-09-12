"""
Parallel (fast) versions of the Monte-Carlo Permutation Tests.

Permutations are completely independent, so they are spread across all CPU
cores with a process pool. Results are IDENTICAL to the serial versions in
mcpt.py because the permutation seeds are drawn in the same order from the
same master RNG -- only the *execution* is parallel, not the math.

Memory note: each worker process receives ONE copy of the price data (via the
pool initializer), not one copy per permutation, so RAM use is roughly
(number of workers) x (size of the dataset).

The originals in mcpt.py are left untouched. See verify_fast.py for the
equivalence proof.

IMPORTANT: the `optimize_func` / `signal_func` you pass must be importable
at module level (defined in a module, not a closure or the __main__ block of
a script) so the process pool can pickle them.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from permutation import get_permutation
from objectives import get_strategy_returns
from mcpt import walkforward_signal


# ---------------------------------------------------------------------------
# In-sample MCPT (parallel)
# ---------------------------------------------------------------------------

# Per-worker globals, populated once by the pool initializer.
_IS_STATE = {}


def _is_init(data, optimize_func):
    _IS_STATE["data"] = data
    _IS_STATE["optimize_func"] = optimize_func


def _is_worker(seed):
    perm = get_permutation(_IS_STATE["data"], start_index=0, seed=int(seed))
    obj, _ = _IS_STATE["optimize_func"](perm)
    return obj


def insample_mcpt_parallel(data, optimize_func, objective_func=None,
                           n_permutations=1000, seed=None, n_jobs=None,
                           verbose=True):
    """
    Parallel in-sample Monte-Carlo Permutation Test.

    Drop-in replacement for mcpt.insample_mcpt. Produces the same p-value
    (given the same `seed`) but runs the permutations across `n_jobs` cores.

    Parameters
    ----------
    data : pd.DataFrame
        OHLC data (Open, High, Low, Close).
    optimize_func : callable
        Module-level function: optimize_func(data) -> (objective, signal).
    objective_func : callable, optional
        Unused except for API symmetry with the serial version.
    n_permutations : int, optional
        Number of permutations. Default 1000.
    seed : int, optional
        Master seed. Same seed -> same result as the serial version.
    n_jobs : int, optional
        Number of worker processes. Default: all CPU cores.
    verbose : bool, optional
        Print a results summary.

    Returns
    -------
    dict with keys: p_value, real_objective, perm_objectives, count_better.
    """
    if n_jobs is None:
        n_jobs = min(os.cpu_count() or 1, 61)  # Windows caps process pools at 61 workers

    # Real optimization (same as serial).
    real_obj, _ = optimize_func(data)

    # Draw permutation seeds in the SAME order as the serial version so the
    # set of permutations -- and therefore the p-value -- is identical.
    rng = np.random.default_rng(seed)
    seeds = [int(rng.integers(0, 2**31)) for _ in range(n_permutations)]

    chunksize = max(1, n_permutations // (n_jobs * 4))
    with ProcessPoolExecutor(max_workers=n_jobs, initializer=_is_init,
                             initargs=(data, optimize_func)) as ex:
        perm_objectives = list(ex.map(_is_worker, seeds, chunksize=chunksize))

    count_better = 1 + sum(1 for o in perm_objectives if o >= real_obj)
    p_value = count_better / (n_permutations + 1)

    if verbose:
        print("--- In-Sample MCPT (parallel) ---")
        print(f"Workers:        {n_jobs}")
        print(f"Real objective: {real_obj:.6f}")
        print(f"Perm mean:      {np.mean(perm_objectives):.6f}")
        print(f"Count >= real:  {count_better} / {n_permutations + 1}")
        print(f"P-value:        {p_value:.4f}")

    return {
        "p_value": p_value,
        "real_objective": real_obj,
        "perm_objectives": perm_objectives,
        "count_better": count_better,
    }


# ---------------------------------------------------------------------------
# Walk-forward MCPT (parallel)
# ---------------------------------------------------------------------------

_WF_STATE = {}


def _wf_init(data, optimize_func, signal_func, objective_func,
             train_bars, train_step):
    _WF_STATE["data"] = data
    _WF_STATE["optimize_func"] = optimize_func
    _WF_STATE["signal_func"] = signal_func
    _WF_STATE["objective_func"] = objective_func
    _WF_STATE["train_bars"] = train_bars
    _WF_STATE["train_step"] = train_step


def _wf_worker(seed):
    train_bars = _WF_STATE["train_bars"]
    perm = get_permutation(_WF_STATE["data"], start_index=train_bars,
                           seed=int(seed))
    sig = walkforward_signal(perm, _WF_STATE["optimize_func"],
                             _WF_STATE["signal_func"], train_bars,
                             _WF_STATE["train_step"])
    rets = get_strategy_returns(sig, perm["Close"].values)
    return _WF_STATE["objective_func"](rets[train_bars:])


def walkforward_mcpt_parallel(data, optimize_func, signal_func, objective_func,
                              train_bars, train_step, n_permutations=200,
                              seed=None, n_jobs=None, verbose=True):
    """
    Parallel walk-forward Monte-Carlo Permutation Test.

    Drop-in replacement for mcpt.walkforward_mcpt. Same p-value (given the
    same `seed`), permutations run across `n_jobs` cores.

    Parameters mirror mcpt.walkforward_mcpt, plus `n_jobs` (default: all cores).

    Returns
    -------
    dict with keys: p_value, real_objective, perm_objectives, real_signal,
    count_better.
    """
    if n_jobs is None:
        n_jobs = min(os.cpu_count() or 1, 61)  # Windows caps process pools at 61 workers

    # Real walk-forward (same as serial).
    real_signal = walkforward_signal(data, optimize_func, signal_func,
                                     train_bars, train_step)
    real_rets = get_strategy_returns(real_signal, data["Close"].values)
    real_obj = objective_func(real_rets[train_bars:])

    rng = np.random.default_rng(seed)
    seeds = [int(rng.integers(0, 2**31)) for _ in range(n_permutations)]

    chunksize = max(1, n_permutations // (n_jobs * 4))
    with ProcessPoolExecutor(
        max_workers=n_jobs, initializer=_wf_init,
        initargs=(data, optimize_func, signal_func, objective_func,
                  train_bars, train_step),
    ) as ex:
        perm_objectives = list(ex.map(_wf_worker, seeds, chunksize=chunksize))

    count_better = 1 + sum(1 for o in perm_objectives if o >= real_obj)
    p_value = count_better / (n_permutations + 1)

    if verbose:
        print("--- Walk-Forward MCPT (parallel) ---")
        print(f"Workers:        {n_jobs}")
        print(f"Real objective: {real_obj:.6f}")
        print(f"Perm mean:      {np.mean(perm_objectives):.6f}")
        print(f"Count >= real:  {count_better} / {n_permutations + 1}")
        print(f"P-value:        {p_value:.4f}")

    return {
        "p_value": p_value,
        "real_objective": real_obj,
        "perm_objectives": perm_objectives,
        "real_signal": real_signal,
        "count_better": count_better,
    }

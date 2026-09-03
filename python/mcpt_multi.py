"""
Multiple-system Monte-Carlo Permutation Test (selection-bias corrected).

The single-strategy MCPT answers: "is THIS strategy's optimized result better
than what optimization finds in noise?" But if you test N strategies and keep
the one that passes, roughly N * alpha of them pass by pure luck. Every extra
strategy you try makes the single test more optimistic.

This test fixes that (Masters' "best of N" / Westfall-Young max-T idea):

  1. Optimize ALL N candidates on the real data      -> real[k], k = 1..N
  2. For each permutation, optimize ALL N candidates -> perm[i, k]
  3. Null distribution of the BEST candidate on noise: best[i] = max_k perm[i, k]
  4. Corrected p-value for candidate k:
         p_corr[k] = (1 + #{ i : best[i] >= real[k] }) / (n_perm + 1)

p_corr asks "could the best-of-N search on noise have produced a result as
good as candidate k?" -- which is exactly the question when you picked k
because it looked best.

Two layers of search are accounted for:
  - within each strategy (its parameter grid) -- because the optimizer runs
    on every permutation, as in the single-strategy MCPT;
  - across strategies (which one you keep)     -- via the max over k.

Requirements: every candidate must use the SAME objective (e.g. profit
factor) so their values are comparable, and the optimize functions must be
module-level / picklable (functools.partial of module-level functions is
fine) for the process pool.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from permutation import get_permutation


_STATE = {}


def _init(data, candidates):
    _STATE["data"] = data
    _STATE["candidates"] = candidates


def _worker(seed):
    perm = get_permutation(_STATE["data"], start_index=0, seed=int(seed))
    return [opt(perm)[0] for _, opt in _STATE["candidates"]]


def multi_system_mcpt(data, candidates, n_permutations=200, seed=None,
                      n_jobs=None, verbose=True):
    """
    Parameters
    ----------
    data : pd.DataFrame
        OHLC data (in-sample / development portion only).
    candidates : dict
        name -> optimize_func, where optimize_func(data) -> (objective, signal).
        Use functools.partial(strategies_library.opt_for_insample,
        optimize=<strategy optimizer>) for library strategies.
    n_permutations : int
    seed : int, optional
    n_jobs : int, optional  (default: all cores)
    verbose : bool

    Returns
    -------
    dict with keys:
        names, real (array), p_uncorrected (array), p_corrected (array),
        best_name, best_real, p_best (corrected p of the best candidate),
        perm (n_permutations x N array), perm_best (n_permutations array)
    """
    if n_jobs is None:
        n_jobs = os.cpu_count()
    names = list(candidates)
    cands = [(k, candidates[k]) for k in names]
    n_cand = len(cands)

    real = np.array([opt(data)[0] for _, opt in cands], dtype=float)

    rng = np.random.default_rng(seed)
    seeds = [int(rng.integers(0, 2**31)) for _ in range(n_permutations)]
    chunksize = max(1, n_permutations // (n_jobs * 4))
    with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init,
                             initargs=(data, cands)) as ex:
        perm = np.array(list(ex.map(_worker, seeds, chunksize=chunksize)),
                        dtype=float).reshape(n_permutations, n_cand)

    perm_best = perm.max(axis=1)
    p_unc = (1 + (perm >= real[None, :]).sum(axis=0)) / (n_permutations + 1)
    p_cor = (1 + (perm_best[:, None] >= real[None, :]).sum(axis=0)) / \
        (n_permutations + 1)

    best_idx = int(np.argmax(real))
    result = {
        "names": names,
        "real": real,
        "p_uncorrected": p_unc,
        "p_corrected": p_cor,
        "best_name": names[best_idx],
        "best_real": float(real[best_idx]),
        "p_best": float(p_cor[best_idx]),
        "perm": perm,
        "perm_best": perm_best,
    }

    if verbose:
        print(f"--- Multiple-system MCPT: {n_cand} candidates, "
              f"{n_permutations} permutations, {n_jobs} workers ---")
        print(f"{'strategy':14s} {'real obj':>9s} {'p (single)':>11s} "
              f"{'p (corrected)':>14s}  verdict")
        for k, name in enumerate(names):
            if p_cor[k] < 0.01:
                verdict = "PASS"
            elif p_unc[k] < 0.01:
                verdict = "passes alone, FAILS after correction"
            else:
                verdict = "fail"
            print(f"{name:14s} {real[k]:9.4f} {p_unc[k]:11.4f} "
                  f"{p_cor[k]:14.4f}  {verdict}")
        print(f"\nBest candidate: {names[best_idx]} "
              f"(real {real[best_idx]:.4f}, corrected p = {p_cor[best_idx]:.4f})")
        print(f"Best-of-{n_cand} on noise: mean {perm_best.mean():.4f}, "
              f"95th pct {np.percentile(perm_best, 95):.4f}")

    return result


if __name__ == "__main__":
    from functools import partial
    from data_loader import load
    from strategies_library import STRATEGIES, opt_for_insample

    df = load("BTC")
    train = df.iloc[:int(len(df) * 0.7)].reset_index(drop=True)
    cands = {name: partial(opt_for_insample, optimize=spec["optimize"])
             for name, spec in STRATEGIES.items()}

    print(f"Development data: BTC, {len(train)} bars\n")
    multi_system_mcpt(train, cands, n_permutations=200, seed=7)

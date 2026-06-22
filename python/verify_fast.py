"""
Equivalence proof: fast versions == original versions.

Runs the original (serial, looped) code and the new (vectorized, parallel)
code on the same data and asserts the results are identical:

  1. Donchian signals match exactly for many lookbacks.
  2. In-sample MCPT p-value matches exactly (same seed).
  3. Walk-forward MCPT p-value matches exactly (same seed).

If this script prints "ALL CHECKS PASSED", the fast code is a safe,
lossless replacement. If anything mismatches, it raises AssertionError.

The optimize/signal wrappers are defined at module level so the parallel
process pool can pickle them.
"""

import time
import numpy as np
import pandas as pd

# Original (slow) implementations
from strategies import donchian_signal, optimize_donchian
from mcpt import insample_mcpt, walkforward_mcpt, walkforward_signal

# Fast implementations
from strategies_fast import donchian_signal_fast, optimize_donchian_fast
from mcpt_fast import insample_mcpt_parallel, walkforward_mcpt_parallel

from objectives import profit_factor

LB = range(5, 40)
TRAIN_BARS = 1000
TRAIN_STEP = 250


def make_data(n=1500, seed=7):
    rng = np.random.default_rng(seed)
    lr = 0.0001 + 0.01 * rng.standard_normal(n)
    close = 100 * np.exp(np.cumsum(lr))
    noise = rng.uniform(0.001, 0.003, n)
    op = close * np.exp(rng.normal(0, 0.001, n))
    hi = np.maximum(op, close) * (1 + noise)
    lo = np.minimum(op, close) * (1 - noise)
    return pd.DataFrame({"Open": op, "High": hi, "Low": lo, "Close": close})


# Module-level wrappers (picklable for the process pool).
def opt_slow(data):
    _, obj, sig = optimize_donchian(data["Close"].values, LB)
    return obj, sig


def opt_fast(data):
    _, obj, sig = optimize_donchian_fast(data["Close"].values, LB)
    return obj, sig


def opt_fast_p(data):
    lb, obj, sig = optimize_donchian_fast(data["Close"].values, LB)
    return obj, sig, lb


def opt_slow_p(data):
    lb, obj, sig = optimize_donchian(data["Close"].values, LB)
    return obj, sig, lb


def sig_fast(data, params):
    return donchian_signal_fast(data["Close"].values, params)


def sig_slow(data, params):
    return donchian_signal(data["Close"].values, params)


def main():
    df = make_data()
    close = df["Close"].values

    print("=" * 60)
    print("CHECK 1: Donchian signals identical across lookbacks")
    print("=" * 60)
    for lb in [5, 12, 20, 33]:
        a = donchian_signal(close, lb)
        b = donchian_signal_fast(close, lb)
        assert np.array_equal(a, b), f"signal mismatch at lookback={lb}"
        print(f"  lookback={lb:>2}: identical ({len(a)} bars)")

    print("\n" + "=" * 60)
    print("CHECK 2: In-sample MCPT p-value identical (same seed)")
    print("=" * 60)
    t = time.time()
    r_slow = insample_mcpt(df, opt_slow, profit_factor,
                           n_permutations=40, seed=1, verbose=False)
    t_slow = time.time() - t
    t = time.time()
    r_fast = insample_mcpt_parallel(df, opt_fast, profit_factor,
                                    n_permutations=40, seed=1, verbose=False)
    t_fast = time.time() - t
    print(f"  serial p-value:   {r_slow['p_value']:.6f}  ({t_slow:.1f}s)")
    print(f"  parallel p-value: {r_fast['p_value']:.6f}  ({t_fast:.1f}s)")
    assert abs(r_slow["p_value"] - r_fast["p_value"]) < 1e-12, "p mismatch"
    assert np.allclose(r_slow["perm_objectives"], r_fast["perm_objectives"]), \
        "perm objective mismatch"
    print("  -> identical p-value and permutation objectives")

    print("\n" + "=" * 60)
    print("CHECK 3: Walk-forward MCPT p-value identical (same seed)")
    print("=" * 60)
    t = time.time()
    w_slow = walkforward_mcpt(df, opt_slow_p, sig_slow, profit_factor,
                              TRAIN_BARS, TRAIN_STEP,
                              n_permutations=20, seed=2, verbose=False)
    t_slow = time.time() - t
    t = time.time()
    w_fast = walkforward_mcpt_parallel(df, opt_fast_p, sig_fast, profit_factor,
                                       TRAIN_BARS, TRAIN_STEP,
                                       n_permutations=20, seed=2, verbose=False)
    t_fast = time.time() - t
    print(f"  serial p-value:   {w_slow['p_value']:.6f}  ({t_slow:.1f}s)")
    print(f"  parallel p-value: {w_fast['p_value']:.6f}  ({t_fast:.1f}s)")
    assert abs(w_slow["p_value"] - w_fast["p_value"]) < 1e-12, "p mismatch"
    assert np.allclose(w_slow["perm_objectives"], w_fast["perm_objectives"]), \
        "perm objective mismatch"
    print("  -> identical p-value and permutation objectives")

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED - fast code is a lossless replacement")
    print("=" * 60)


if __name__ == "__main__":
    main()

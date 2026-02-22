"""
Monte-Carlo Permutation Tests (MCPT) for trading strategy validation.

Implements the two key tests from the YouTuber's 4-step framework:
  1. In-Sample MCPT  - Tests if in-sample performance is due to real patterns
                       or just data-mining bias from optimization.
  2. Walk-Forward MCPT - Tests if walk-forward results are due to learned
                         patterns generalizing, or just dumb luck.

Based on the methods from "Testing and Tuning Market Trading Systems"
by Timothy Masters.
"""

import numpy as np
from permutation import get_permutation
from objectives import get_strategy_returns


def insample_mcpt(data, optimize_func, objective_func, n_permutations=1000,
                  seed=None, verbose=True):
    """
    In-Sample Monte-Carlo Permutation Test.

    Tests the null hypothesis: "The strategy's in-sample performance is
    entirely due to data-mining bias from optimization."

    Algorithm:
      1. Optimize the strategy on real data -> real_objective
      2. For each permutation:
         a. Generate a permutation of the price data
         b. Optimize the strategy on the permuted data -> perm_objective
         c. If perm_objective >= real_objective, increment counter
      3. p_value = counter / (n_permutations + 1)

    A low p-value (< 0.01) means the strategy likely has real predictive
    power beyond what optimization alone could find in noise.

    Parameters
    ----------
    data : pd.DataFrame
        OHLC data with columns: Open, High, Low, Close.
    optimize_func : callable
        Function that takes a DataFrame and returns (objective_value, signal).
        This should optimize the strategy on the given data and return the
        best objective value found.
        Signature: optimize_func(data) -> (float, np.ndarray)
    objective_func : callable
        The objective function used (e.g., profit_factor). Only used
        for labeling in verbose output.
    n_permutations : int, optional
        Number of permutations to run. Default is 1000.
        Minimum recommended: 100. Ideal: 1000+.
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool, optional
        Print progress and results. Default is True.

    Returns
    -------
    dict
        Results containing:
        - 'p_value': quasi p-value
        - 'real_objective': objective on real data
        - 'perm_objectives': list of objectives on permutations
        - 'count_better': number of permutations >= real objective
    """
    rng = np.random.default_rng(seed)

    # Step 1: Optimize on real data
    real_obj, real_signal = optimize_func(data)
    if verbose:
        print(f"Real data objective: {real_obj:.6f}")

    # Step 2: Permutation loop
    count_better = 1  # Count the original in the results (as per Masters)
    perm_objectives = []

    for i in range(n_permutations):
        perm_seed = rng.integers(0, 2**31)
        perm_data = get_permutation(data, start_index=0, seed=perm_seed)

        perm_obj, _ = optimize_func(perm_data)
        perm_objectives.append(perm_obj)

        if perm_obj >= real_obj:
            count_better += 1

        if verbose and (i + 1) % 50 == 0:
            current_p = count_better / (i + 2)
            print(f"  Permutation {i + 1}/{n_permutations} | "
                  f"current p-value: {current_p:.4f}")

    # Step 3: Compute p-value
    p_value = count_better / (n_permutations + 1)

    if verbose:
        print(f"\n--- In-Sample MCPT Results ---")
        print(f"Real objective:    {real_obj:.6f}")
        print(f"Perm mean:         {np.mean(perm_objectives):.6f}")
        print(f"Perm std:          {np.std(perm_objectives):.6f}")
        print(f"Count >= real:     {count_better} / {n_permutations + 1}")
        print(f"P-value:           {p_value:.4f}")
        if p_value < 0.01:
            print("Result: PASS - Strong evidence of real predictive power.")
        elif p_value < 0.05:
            print("Result: MARGINAL - Some evidence, but not convincing.")
        else:
            print("Result: FAIL - Performance likely due to data-mining bias.")

    return {
        "p_value": p_value,
        "real_objective": real_obj,
        "perm_objectives": perm_objectives,
        "count_better": count_better,
    }


def walkforward_signal(data, optimize_func, signal_func,
                       train_bars, train_step):
    """
    Generate a walk-forward signal by periodically re-optimizing.

    At each re-optimization point, the strategy is optimized on the
    most recent `train_bars` of data, then traded forward for
    `train_step` bars until the next re-optimization.

    Parameters
    ----------
    data : pd.DataFrame
        Full OHLC data (training + out-of-sample).
    optimize_func : callable
        Function: optimize_func(data) -> (objective, signal, params)
        Returns the best objective, signal, and parameters.
    signal_func : callable
        Function: signal_func(data, params) -> signal
        Generates a signal for given data using specific parameters.
    train_bars : int
        Number of bars in the training window.
    train_step : int
        Number of bars between re-optimizations.

    Returns
    -------
    np.ndarray
        Walk-forward signal for the entire dataset.
        Bars before the first training window end will be 0.
    """
    n = len(data)
    wf_signal = np.zeros(n)

    # First optimization point: end of first training window
    next_train = train_bars

    current_params = None

    while next_train <= n:
        # Optimize on training window
        train_data = data.iloc[next_train - train_bars:next_train].copy()
        train_data = train_data.reset_index(drop=True)
        _, _, current_params = optimize_func(train_data)

        # Apply signal from this optimization point forward
        oos_end = min(next_train + train_step, n)

        if next_train < n:
            oos_data = data.iloc[:oos_end].copy()
            oos_data = oos_data.reset_index(drop=True)
            full_signal = signal_func(oos_data, current_params)
            wf_signal[next_train:oos_end] = full_signal[next_train:oos_end]

        next_train += train_step

    return wf_signal


def walkforward_mcpt(data, optimize_func, signal_func, objective_func,
                     train_bars, train_step, n_permutations=200,
                     seed=None, verbose=True):
    """
    Walk-Forward Monte-Carlo Permutation Test.

    Tests the null hypothesis: "The walk-forward results could have been
    achieved by a worthless strategy through dumb luck."

    Algorithm:
      1. Generate walk-forward signal on real data -> real_objective
      2. For each permutation:
         a. Permute ONLY the data after the first training window
            (start_index = train_bars)
         b. Generate walk-forward signal on permuted data -> perm_objective
         c. If perm_objective >= real_objective, increment counter
      3. p_value = counter / (n_permutations + 1)

    Key insight: We only permute the out-of-sample portion. The training
    data stays real so the strategy can still optimize. But the OOS data
    is randomized, so any patterns the strategy "learned" won't be there.

    Parameters
    ----------
    data : pd.DataFrame
        Full OHLC data (must include training + OOS periods).
    optimize_func : callable
        Function: optimize_func(data) -> (objective, signal, params)
    signal_func : callable
        Function: signal_func(data, params) -> signal
    objective_func : callable
        Function: objective_func(returns) -> float
    train_bars : int
        Number of bars in the training window.
    train_step : int
        Bars between re-optimizations (e.g., 30*24 for 30 days of hourly).
    n_permutations : int, optional
        Number of permutations. Default is 200 (this test is slow).
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool, optional
        Print progress. Default is True.

    Returns
    -------
    dict
        Results containing:
        - 'p_value': quasi p-value
        - 'real_objective': objective on real walk-forward
        - 'perm_objectives': list of permutation objectives
        - 'real_signal': the walk-forward signal on real data
    """
    rng = np.random.default_rng(seed)

    # Step 1: Walk-forward on real data
    real_signal = walkforward_signal(data, optimize_func, signal_func,
                                     train_bars, train_step)
    real_returns = get_strategy_returns(real_signal, data["Close"].values)

    # Only evaluate OOS portion (after first training window)
    oos_returns = real_returns[train_bars:]
    real_obj = objective_func(oos_returns)

    if verbose:
        print(f"Real walk-forward objective: {real_obj:.6f}")

    # Step 2: Permutation loop
    count_better = 1
    perm_objectives = []

    for i in range(n_permutations):
        perm_seed = rng.integers(0, 2**31)

        # Only permute data AFTER the first training window
        perm_data = get_permutation(data, start_index=train_bars,
                                    seed=perm_seed)

        perm_signal = walkforward_signal(perm_data, optimize_func,
                                         signal_func, train_bars, train_step)
        perm_returns = get_strategy_returns(perm_signal,
                                            perm_data["Close"].values)
        perm_oos_returns = perm_returns[train_bars:]
        perm_obj = objective_func(perm_oos_returns)
        perm_objectives.append(perm_obj)

        if perm_obj >= real_obj:
            count_better += 1

        if verbose and (i + 1) % 10 == 0:
            current_p = count_better / (i + 2)
            print(f"  Permutation {i + 1}/{n_permutations} | "
                  f"current p-value: {current_p:.4f}")

    # Step 3: Compute p-value
    p_value = count_better / (n_permutations + 1)

    if verbose:
        print(f"\n--- Walk-Forward MCPT Results ---")
        print(f"Real WF objective: {real_obj:.6f}")
        print(f"Perm mean:         {np.mean(perm_objectives):.6f}")
        print(f"Perm std:          {np.std(perm_objectives):.6f}")
        print(f"Count >= real:     {count_better} / {n_permutations + 1}")
        print(f"P-value:           {p_value:.4f}")
        if p_value < 0.01:
            print("Result: PASS - Walk-forward results are statistically "
                  "significant.")
        elif p_value < 0.05:
            print("Result: MARGINAL - Some evidence, needs more OOS data.")
        else:
            print("Result: FAIL - Walk-forward results could be dumb luck.")

    return {
        "p_value": p_value,
        "real_objective": real_obj,
        "perm_objectives": perm_objectives,
        "real_signal": real_signal,
        "count_better": count_better,
    }

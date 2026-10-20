"""Unit tests for ExperimentLab statistical core.

Run:  pytest experimentlab/tests -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import demo_data, stats_core as sc  # noqa: E402


# ----------------------------------------------------------------- power ----

def test_sample_size_matches_textbook_example():
    # p1=0.10, p2=0.12 (20% relative lift), alpha=.05, power=.80.
    # R: power.prop.test(p1=0.10, p2=0.12, power=0.80) => n = 3840.9/arm.
    n = sc.required_sample_size(0.10, 0.20, alpha=0.05, power=0.80)
    assert 3700 <= n <= 4000


def test_larger_mde_needs_fewer_users():
    n_small_mde = sc.required_sample_size(0.10, 0.02)
    n_big_mde = sc.required_sample_size(0.10, 0.20)
    assert n_big_mde < n_small_mde


def test_higher_power_needs_more_users():
    assert (sc.required_sample_size(0.10, 0.10, power=0.90)
            > sc.required_sample_size(0.10, 0.10, power=0.80))


def test_achieved_power_at_required_n_hits_target():
    n = sc.required_sample_size(0.05, 0.10, alpha=0.05, power=0.80)
    assert sc.achieved_power(0.05, 0.10, n, alpha=0.05) >= 0.79


def test_power_monotone_in_n():
    powers = [sc.achieved_power(0.10, 0.05, n) for n in (1000, 5000, 20000)]
    assert powers == sorted(powers)


def test_invalid_rate_raises():
    with pytest.raises(ValueError):
        sc.required_sample_size(1.5, 0.1)   # baseline outside (0, 1)
    with pytest.raises(ValueError):
        sc.required_sample_size(0.9, 0.5)   # p2 = 1.35 > 1


# ------------------------------------------------------------------- srm ----

def test_srm_clean_split_not_significant():
    p = sc.srm_p_value({"control": 5000, "treatment": 5010},
                       {"control": 1, "treatment": 1})
    assert p > 0.5


def test_srm_skewed_split_detected():
    p = sc.srm_p_value({"control": 5000, "treatment": 5600},
                       {"control": 1, "treatment": 1})
    assert p < 1e-6


def test_srm_three_arms_weighted_expected():
    # 60/30/10 expected, observed matches
    p = sc.srm_p_value({"a": 6000, "b": 3000, "c": 1000},
                       {"a": 6, "b": 3, "c": 1})
    assert p > 0.5


# ----------------------------------------------------------- frequentist ----

def test_z_test_recovers_known_effect():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "variant": ["control"] * 4000 + ["treatment"] * 4000,
        "converted": np.concatenate([
            rng.binomial(1, 0.10, 4000),
            rng.binomial(1, 0.13, 4000),
        ]),
    })
    res = sc.frequentist_test(df, "variant", "converted",
                              "control", "treatment")
    assert res.method == "Two-proportion z-test"
    assert 0.02 < res.absolute_uplift < 0.04
    assert res.p_value < 0.05 and res.significant
    assert res.ci_low > 0  # CI excludes zero when significant


def test_z_test_null_is_not_significant():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "variant": ["control"] * 3000 + ["treatment"] * 3000,
        "converted": np.concatenate([
            rng.binomial(1, 0.10, 3000),
            rng.binomial(1, 0.10, 3000),
        ]),
    })
    res = sc.frequentist_test(df, "variant", "converted",
                              "control", "treatment")
    assert not res.significant
    assert res.ci_low < 0 < res.ci_high


def test_welch_matches_scipy_reference():
    rng = np.random.default_rng(2)
    a, b = rng.normal(4, 2, 500), rng.normal(4.6, 3, 500)
    df = pd.DataFrame({
        "variant": ["control"] * 500 + ["treatment"] * 500,
        "revenue": np.concatenate([a, b]),
    })
    res = sc.frequentist_test(df, "variant", "revenue", "control", "treatment",
                              metric_type="continuous", method="t")
    ref = stats.ttest_ind(b, a, equal_var=False)
    assert res.p_value == pytest.approx(ref.pvalue, rel=1e-10)
    assert res.absolute_uplift == pytest.approx(b.mean() - a.mean())


def test_conversion_with_small_cells_raises():
    df = pd.DataFrame({
        "variant": ["control", "treatment"],
        "converted": [1, 0],
    })
    with pytest.raises(ValueError, match="Fisher"):
        sc.frequentist_test(df, "variant", "converted", "control", "treatment")


# -------------------------------------------------------------- bayesian ----

def test_bayesian_clear_winner_high_probability():
    res = sc.bayesian_test(x_a=100, n_a=1000, x_b=150, n_b=1000,
                           n_samples=50_000)
    assert res.prob_b_beats_a > 0.97
    assert res.risk < 0.01


def test_bayesian_symmetric_prior_near_half():
    res = sc.bayesian_test(x_a=100, n_a=1000, x_b=100, n_b=1000,
                           n_samples=50_000)
    assert 0.40 < res.prob_b_beats_a < 0.60


def test_bayesian_ci_brackets_expected_uplift():
    res = sc.bayesian_test(x_a=100, n_a=1000, x_b=130, n_b=1000,
                           n_samples=50_000)
    assert res.ci_low < res.expected_uplift < res.ci_high


# ------------------------------------------------------------------ cuped ----

def test_cuped_reduces_variance_when_correlated():
    rng = np.random.default_rng(3)
    x = rng.normal(10, 2, 4000)
    y = 10 + 0.6 * (x - 10) + rng.normal(0, 1.5, 4000)
    y_adj = sc.cuped_adjustment(y, x)
    assert sc.variance_reduction(y, y_adj) > 0.25


def test_cuped_preserves_overall_mean():
    rng = np.random.default_rng(4)
    x = rng.normal(10, 2, 4000)
    y = 0.5 * x + rng.normal(0, 1, 4000)
    y_adj = sc.cuped_adjustment(y, x)
    assert y_adj.mean() == pytest.approx(y.mean(), rel=1e-9)


def test_cuped_zero_variance_covariate_is_noop():
    y = np.array([1.0, 2.0, 3.0])
    x = np.array([5.0, 5.0, 5.0])
    np.testing.assert_array_equal(sc.cuped_adjustment(y, x), y)


def test_cuped_on_demo_data_reduces_variance():
    df = demo_data.cuped_experiment(rho=0.6)
    y_adj = sc.cuped_adjustment(df["metric"].to_numpy(),
                                df["pre_metric"].to_numpy())
    assert sc.variance_reduction(df["metric"].to_numpy(), y_adj) > 0.2


# ------------------------------------------------------------ demo data -----

def test_demo_conversion_shape_and_determinism():
    a = demo_data.conversion_experiment(n_per_arm=500, seed=5)
    b = demo_data.conversion_experiment(n_per_arm=500, seed=5)
    pd.testing.assert_frame_equal(a, b)
    assert set(a["variant"]) == {"control", "treatment"}
    assert a["converted"].isin([0, 1]).all()


def test_demo_srm_shift_changes_split():
    df = demo_data.conversion_experiment(n_per_arm=5000, srm_shift=0.05, seed=1)
    share_t = (df["variant"] == "treatment").mean()
    assert 0.53 < share_t < 0.57


# ------------------------------------------------- text-column regression ----

def test_text_metric_raises_helpful_error_not_crash():
    df = demo_data.conversion_experiment(n_per_arm=500, seed=5)
    with pytest.raises(ValueError, match="not 0/1"):
        sc.frequentist_test(df, "variant", "device", "control", "treatment",
                            metric_type="conversion")
    with pytest.raises(ValueError, match="numeric"):
        sc.frequentist_test(df, "variant", "device", "control", "treatment",
                            metric_type="continuous", method="t")


def test_binary_and_float_helpers():
    df = demo_data.conversion_experiment(n_per_arm=200, seed=5)
    assert sc._is_binary(df["converted"])
    assert not sc._is_binary(df["device"])
    num = sc._to_float(df["converted"], "x")
    assert num.sum() == df["converted"].sum()
    with pytest.raises(ValueError, match="looks like text"):
        sc._to_float(df["device"], "x")

